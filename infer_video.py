#!/usr/bin/env python3
"""
MiniCPM-V-4.5 原生视频推理脚本
支持长视频自动分段（默认10秒一段），模型只加载一次，结果自动整合。
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoModel, AutoTokenizer

# Patch: transformers 5.7.0 要求 all_tied_weights_keys，但 MiniCPM-V-4.5 旧代码只有 _tied_weights_keys
import transformers.modeling_utils as _modeling_utils
if hasattr(_modeling_utils.PreTrainedModel, "_move_missing_keys_from_meta_to_device"):
    _original_move = _modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _original_move(self, *args, **kwargs)

    _modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

# ========== 视频编码（来自官方 README） ==========
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3

# 3090 24GB 显存安全上限（visual encoder 显存瓶颈）
DEFAULT_MAX_TOTAL_FRAMES = 24
TIME_SCALE = 0.1


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    return len(vr) / fps


def encode_video(video_path, choose_fps=3, force_packing=None,
                 max_total_frames=DEFAULT_MAX_TOTAL_FRAMES,
                 start_time=0.0, end_time=None):
    """从视频中提取帧，支持指定时间段"""
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    if end_time is None:
        end_time = video_duration

    segment_duration = end_time - start_time
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), len(vr))
    frame_idx_range = list(range(start_frame, end_frame))

    if len(frame_idx_range) == 0:
        return [], [], segment_duration

    if choose_fps * int(segment_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, segment_duration))
    else:
        packing_nums = math.ceil(segment_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(segment_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    # 显存保护：强制限制总帧数
    if choose_frames > max_total_frames:
        choose_frames = max_total_frames
        packing_nums = math.ceil(choose_frames / MAX_NUM_FRAMES)
        if packing_nums < 1:
            packing_nums = 1

    choose_frames = min(choose_frames, len(frame_idx_range))
    if choose_frames <= 0:
        return [], [], segment_duration
    frame_idx = np.array(uniform_sample(frame_idx_range, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)

    print(f"[*] 段内时长: {segment_duration:.1f}s, 原始FPS: {fps:.1f}")
    print(f"[*] 采样帧数: {len(frame_idx)}, packing_nums: {packing_nums}")

    frames = vr.get_batch(frame_idx).asnumpy()

    # temporal_ids 使用段内相对时间
    frame_idx_ts = (frame_idx - start_frame) / fps
    scale = np.arange(0, segment_duration, TIME_SCALE)
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)

    return frames, frame_ts_id_group, segment_duration


# ========== Prompt 构建 ==========
def load_action_descriptions():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    desc_path = os.path.join(project_root, "action_descriptions.json")
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_prompt(actions, duration, use_detailed=False):
    action_desc = load_action_descriptions() if use_detailed else {}

    action_lines = []
    for i, action in enumerate(actions, 1):
        if use_detailed and action in action_desc:
            desc = action_desc[action]
            start_desc = desc.get('start', '未定义')
            end_desc = desc.get('end', '未定义')
            action_lines.append(
                f"{i}. {action}\n"
                f"   【开始判定标准】{start_desc}\n"
                f"   【结束判定标准】{end_desc}"
            )
        else:
            default_desc = {
                "追球": "狗奔跑追逐球/物体的动作",
                "拉球": "狗咬住球后，身体向后发力拉扯的动作",
                "咬球": "狗张嘴试图咬住球的动作",
                "犬追物体": "狗奔跑追逐物体的动作",
                "犬拉": "狗咬住物品后身体向后发力拉扯的动作",
                "犬尝试咬物体": "狗张嘴试图咬住物体的动作",
                "犬咬住物体拉动": "狗嘴里含有物体并主动拉拽的动作",
                "犬咬住物体持续咬住": "物体在狗嘴里被闭合包住的持住状态",
                "犬咬": "狗咬住训导员手中/牵引的物体，嘴闭合包住，正在咀嚼或持续含住的状态",
            }
            desc = default_desc.get(action, "")
            if desc:
                action_lines.append(f"{i}. {action}：{desc}")
            else:
                action_lines.append(f"{i}. {action}")

    actions_text = "\n".join(action_lines)

    json_examples = [f'"{action}": {{"start": X, "end": Y}}' for action in actions]
    json_example_str = ", ".join(json_examples)

    prompt = (
        f"你是一个视频动作分析专家。下面是一个约 {duration:.0f} 秒的狗视频片段。\n\n"
        f"请**严格按以下三个步骤**分析，每步都必须用文字写出，最后输出 JSON。\n\n"
        f"【步骤1】观察并描述球的位置和狗的动作\n\n"
        f"请明确写出一句话回答：\n"
        f"- 球在哪里？\n"
        f"  - 在狗的嘴里（已叼住）\n"
        f"  - 在地上（狗未接触）\n"
        f"  - 在空中/被抛出（狗正在追逐）\n"
        f"  - 被训导员拿在手里\n"
        f"  - 完全没有看到球\n\n"
        f"**重要：如果完全没有看到球，必须明确写出‘没有看到球’。**\n\n"
        f"然后，用一两句话描述狗正在做什么（例如：奔跑、慢走、站立、嗅闻、跳跃、空咬等）。\n\n"
        f"【步骤2】判定动作是否出现（必须写文字 + 理由）\n\n"
        f"你需要从下面列表中判断哪些动作**确实出现**：\n"
        f"{actions_text}\n\n"
        f"每个动作的定义如下。**注意：每个动作的第一条属性是‘先决条件’，如果不满足，该动作直接判‘未出现’，无需检查其他属性。**\n\n"
        f"## 追球\n"
        f"- **先决条件**：球存在（不是‘完全没有看到球’）且球**不在**狗嘴里。\n"
        f"- **属性**：狗处于奔跑或快速移动状态；身体姿态向前、兴奋（尾巴翘起、耳朵竖起等）。\n"
        f"- **描述**：狗从静止或慢走开始加速，朝着球的方向快速奔跑，期间球始终未被叼住。\n"
        f"- **区分**：与拉球的区别在于没有咬住后的拉扯；与犬咬的区别在于球未在嘴里。\n\n"
        f"## 拉球\n"
        f"- **先决条件**：球存在且球**已经在狗嘴里**。\n"
        f"- **属性**：狗身体向后倾斜、四肢抓地、头部和颈部向后用力；有明显的拉扯动作。\n"
        f"- **描述**：狗咬住球后，通过后撤身体、扭动头部或甩动颈部来对抗阻力。\n"
        f"- **区分**：仅嘴里有球但不向后发力不算拉球。\n\n"
        f"## 犬咬\n"
        f"- **先决条件**：球存在且球**完全在狗嘴里**（狗闭嘴含住或用牙齿咬住）。\n"
        f"- **属性**：球在口中；可能伴随咀嚼或叼着走，但没有向后拉扯的发力动作。\n"
        f"- **描述**：狗用嘴含住或咬住球，保持球在口中，可以静止、行走或轻微摇头。\n"
        f"- **区分**：犬咬是一种**状态**（球在嘴里），拉球是一种**动态动作**。\n\n"
        f"**判定铁律（必须逐条遵守）：**\n"
        f"1. **先决条件优先**：如果步骤1中球的位置是‘完全没有看到球’，则追球、拉球、犬咬**全部**直接判‘未出现’。\n"
        f"2. **证据必须来自步骤1**：步骤1中没提到的视觉信息，不能作为判定依据。\n"
        f"3. **动作必须实际完成**：准备、试图、快要做了都不算。\n"
        f"4. 如果步骤1的描述与任何属性矛盾，该动作直接判‘未出现’。\n"
        f"5. 如果狗的行为不属于上述任何动作，请明确写‘不属于上述动作’，并描述实际行为。\n\n"
        f"**请按以下格式写出判定文字（必须首先写出步骤1中关于球的结论）：**\n"
        f"- 步骤1结论：球的位置是 [在嘴里/在地上/在空中/被拿着/没有看到球]。\n"
        f"- 追球：出现/未出现。理由：先决条件是否满足？属性是否满足？……\n"
        f"- 拉球：出现/未出现。理由：……\n"
        f"- 犬咬：出现/未出现。理由：……\n\n"
        f"【步骤3】给出动作的时间区间（仅当动作出现时）\n\n"
        f"**时间计算规则：**\n"
        f"- 起始时间 = 本片段开头的 0 秒\n"
        f"- 结束时间 ≤ {duration:.0f} 秒\n"
        f"- **完全忽略画面上的水印时间戳**（如 15:03:38）\n"
        f"- 根据实际动作的起止帧判断：例如追球从第 2 秒加速奔跑开始，到第 3.5 秒停下或咬到球为止。\n\n"
        f"**先写文字说明每个动作的起止时间：**\n"
        f"- 追球：2 秒到 3.5 秒\n"
        f"- 犬咬：未出现\n"
        f"- 拉球：4 秒到 5 秒\n\n"
        f"最后输出 JSON（必须包含在回答末尾）：\n\n"
        f"```json\n"
        f"{json_example_str}\n"
        f"```\n"
        f"未出现的动作，start 和 end 都填 null。\n"
    )
    return prompt


def extract_json(text):
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def run_single_segment(model, tokenizer, video_path, actions, seg_start, seg_end,
                       fps, use_detailed, device):
    """运行单段推理，返回修正后的 parsed_result 和 raw_output"""
    frames, temporal_ids, seg_duration = encode_video(
        video_path, choose_fps=fps,
        start_time=seg_start, end_time=seg_end
    )

    if len(frames) == 0:
        print(f"[!] 段 {seg_start:.1f}s-{seg_end:.1f}s 未能提取到帧，跳过")
        return None, ""

    prompt = build_prompt(actions, seg_duration, use_detailed=use_detailed)
    msgs = [{'role': 'user', 'content': frames + [prompt]}]

    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        use_image_id=False,
        max_slice_nums=1,
        temporal_ids=temporal_ids,
        do_sample=False
    )

    result_text = answer if isinstance(answer, str) else str(answer)
    parsed = extract_json(result_text)

    # 修正时间偏移
    if parsed:
        for action in actions:
            if action in parsed and isinstance(parsed[action], dict):
                if parsed[action].get('start') is not None:
                    parsed[action]['start'] = round(parsed[action]['start'] + seg_start, 1)
                if parsed[action].get('end') is not None:
                    parsed[action]['end'] = round(parsed[action]['end'] + seg_start, 1)

    return parsed, result_text




# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 视频动作分析（支持长视频自动分段）")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--actions", required=True, help="动作列表，逗号分隔")
    parser.add_argument("--model", default=None, help="模型本地路径（默认自动查找）")
    parser.add_argument("--fps", type=int, default=3, help="视频采样fps (默认3)")
    parser.add_argument("--detailed", action="store_true", help="加载 action_descriptions.json 详细定义")
    parser.add_argument("--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（极慢，仅测试）")
    parser.add_argument("--segment-duration", type=float, default=10,
                        help="分段时长（秒），超过则自动分段。默认10")
    parser.add_argument("--start-time", type=float, default=None,
                        help="手动指定起始时间（秒），与 --end-time 配合只跑一段")
    parser.add_argument("--end-time", type=float, default=None,
                        help="手动指定结束时间（秒），与 --start-time 配合只跑一段")
    args = parser.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    if not actions:
        print("[错误] 请至少指定一个动作")
        sys.exit(1)

    # 自动查找模型路径
    model_path = args.model
    if model_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "MiniCPM-V-4_5"),
            os.path.join(os.path.dirname(__file__), "..", "coarse_locator", "MiniCPM-V-4_5"),
            os.path.join(os.path.dirname(__file__), "..", "MiniCPM-V-4_5"),
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(os.path.join(c, "config.json")):
                model_path = c
                break

    if not model_path or not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[错误] 找不到模型配置文件，请通过 --model 指定路径")
        sys.exit(1)

    # 获取视频时长
    duration = get_video_duration(args.video)

    print("=" * 60)
    print("MiniCPM-V-4.5 原生视频推理")
    print("=" * 60)
    print(f"[*] 视频: {args.video}")
    print(f"[*] 时长: {duration:.1f}s")
    print(f"[*] 动作: {actions}")
    print(f"[*] 模型: {model_path}")
    print(f"[*] 采样: {args.fps} fps")
    print(f"[*] 分段: {args.segment_duration}s")
    print(f"[*] 详细定义: {'是' if args.detailed else '否'}")
    print(f"[*] 设备: {'CPU' if args.cpu else 'CUDA (自动)'}")

    # 加载模型（只加载一次）
    print(f"\n[*] 正在加载模型...")
    device = "cpu" if args.cpu else "cuda"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False
    )
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[*] 模型加载完成，设备: {device}")

    # 判断是否手动指定时间段
    if args.start_time is not None and args.end_time is not None:
        print(f"\n[*] 手动指定时间段: {args.start_time:.1f}s - {args.end_time:.1f}s")
        parsed, raw = run_single_segment(
            model, tokenizer, args.video, actions,
            args.start_time, args.end_time, args.fps, args.detailed, device
        )
        if raw:
            print(f"[*] 模型输出:\n{raw}\n")
        if parsed:
            print(f"[*] 结果: {json.dumps(parsed, ensure_ascii=False)}")
        all_raw = [raw] if raw else []
        all_parsed = [parsed] if parsed else []
    elif duration <= args.segment_duration:
        print(f"\n[*] 视频时长 {duration:.1f}s <= 分段阈值，直接推理...")
        parsed, raw = run_single_segment(
            model, tokenizer, args.video, actions,
            0, duration, args.fps, args.detailed, device
        )
        all_raw = [raw] if raw else []
        all_parsed = [parsed] if parsed else []
    else:
        num_segments = math.ceil(duration / args.segment_duration)
        print(f"\n[*] 视频时长 {duration:.1f}s > 分段阈值，自动分为 {num_segments} 段推理...")

        all_raw = []
        all_parsed = []
        for i in range(num_segments):
            seg_start = i * args.segment_duration
            seg_end = min((i + 1) * args.segment_duration, duration)
            print(f"\n{'='*60}")
            print(f"[*] 正在处理第 {i+1}/{num_segments} 段: {seg_start:.1f}s - {seg_end:.1f}s")
            print(f"{'='*60}")

            parsed, raw = run_single_segment(
                model, tokenizer, args.video, actions,
                seg_start, seg_end, args.fps, args.detailed, device
            )

            if raw:
                all_raw.append(f"=== 段 {i+1} ({seg_start:.1f}s-{seg_end:.1f}s) ===\n{raw}")
                print(f"[*] 模型输出:\n{raw}\n")
            if parsed:
                all_parsed.append(parsed)
                print(f"[*] 段结果: {json.dumps(parsed, ensure_ascii=False)}")

    # 输出各段结果（不做合并，直接输出每段原始结果）
    print("\n" + "=" * 60)
    print("各段结果:")
    print("=" * 60)
    for i, p in enumerate(all_parsed):
        print(f"段 {i+1}: {json.dumps(p, ensure_ascii=False)}")

    # 保存结果
    output_data = {
        "video": os.path.abspath(args.video),
        "model": model_path,
        "fps": args.fps,
        "duration": round(duration, 2),
        "segment_duration": args.segment_duration,
        "actions": actions,
        "use_detailed": args.detailed,
        "raw_outputs": "\n\n".join(all_raw),
        "segment_results": all_parsed
    }

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存: {out_path}")
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        default_out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "minicpm_v45", f"{video_name}_v45.json"
        )
        os.makedirs(os.path.dirname(default_out), exist_ok=True)
        with open(default_out, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存: {default_out}")

    print("\n[*] 完成")


if __name__ == "__main__":
    main()
