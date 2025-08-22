# -*- coding: utf-8 -*-
# @Author  : wenshao & siya
# @Email   : wenshaoguo1026@gmail.com & siyayan1222@gmail.com
# @Project : WebLivePortrait
# @FileName: webui.py
import gradio as gr
import os.path as osp
from omegaconf import OmegaConf

from collections import deque
import threading, time
import queue

import cv2
import numpy as np

import argparse



from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline 


parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
parser.add_argument('--mode', required=False, type=str, default="trt")
parser.add_argument('--use_mp', action='store_true', help='use mediapipe or not')
parser.add_argument(
    "--host_ip", type=str, default="0.0.0.0", help="host ip"
)
parser.add_argument("--port", type=int, default=9870, help="server port")
args, unknown = parser.parse_known_args()

if args.mode == "onnx":
    cfg_path = "configs/onnx_mp_infer.yaml" if args.use_mp else "configs/onnx_infer.yaml"
else:
    cfg_path = "configs/trt_mp_infer.yaml" if args.use_mp else "configs/trt_infer.yaml"
    
infer_cfg = OmegaConf.load(cfg_path)
gradio_pipeline = GradioLivePortraitPipeline(infer_cfg)
live_pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)  # uses same config as app  # :contentReference[oaicite:5]{index=5}


def gpu_wrapped_execute_video(*args, **kwargs):
    return gradio_pipeline.execute_video(*args, **kwargs)


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


def change_animal_model(is_animal):
    global gradio_pipeline
    gradio_pipeline.clean_models()
    gradio_pipeline.init_models(is_animal=is_animal)

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

TARGET_W, TARGET_H = 512, 512
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"

eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(format="png", type="numpy")
output_image_paste_back = gr.Image(format="png", type="numpy")

js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
        
        // --- set webcam constraints:  @ 8fps ---
        navigator.mediaDevices.getUserMedia({
            video: { width:512, height:512, frameRate: { ideal: 8, max: 8 } }
        }).then(stream => {
            // Find the first video element created by gradio webcam
            const video = document.querySelector("video");
            if (video) {
                video.srcObject = stream;
            }
        }).catch(err => {
            console.error("Webcam constraint error:", err);
        });
    }
    """
    

def ensure_worker(st):
    """Start the worker once per session."""
    t = st.get("worker_thread")
    if t and t.is_alive():
        return st

    st["stop_event"] = threading.Event()
    st["frame_buf"] = queue.Queue(maxsize=10)
    st["result_buf"] = deque(maxlen=10)
    st["first_frame"] = True
    
    t = threading.Thread(target=infer_worker, args=(st,), daemon=True)
    t.start()
    st["worker_thread"] = t
    st["worker_started"] = True
    return st

def infer_worker(st):
    stop_event = st["stop_event"]
    first = True
    while not stop_event.is_set():
        try:
            # wait for next frame (blocks until producer puts one)
            driving_frame_rgb = st["frame_buf"].get(timeout=1.0)
        except queue.Empty:
            continue

        if not st.get("src_ready"):
            continue


        img_bgr = cv2.cvtColor(driving_frame_rgb, cv2.COLOR_RGB2BGR)

        if stop_event.is_set():   # cooperative cancel
            break
        # --- your heavy pipeline ---
        img_crop, out_crop, out_org, _ = live_pipe.run(
            img_bgr, live_pipe.src_imgs[0], live_pipe.src_infos[0], first_frame=first
        )
        first = False

        if out_crop is None and out_org is None:
            # keep last result if any
            pass
        else:
            vis_rgb = cv2.resize(out_org, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            st["result_buf"].append(vis_rgb)
    # optional: cleanup live_pipe here
    print("worker stopped")
    st["running"] = False


def _draw_fps(rgb, st):
    if rgb is None:
        return None  # don’t crash
    
    frame = rgb.copy()
    fps = st.get("fps", 0.0)
    text = f"{fps:.1f} FPS"
    h, w = frame.shape[:2]

    # draw red text, bottom-right corner
    cv2.putText(
        frame, text,
        (w - 150, h - 20),  # shift inward so it’s visible
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2, cv2.LINE_AA
    )
    return frame
    
def live_step(driving_frame_rgb, st):
    if driving_frame_rgb is None:
        return None, st

    now = time.time()
    prev = st.get("last_time", now)
    dt = now - prev
    if dt > 0:
        st["fps"] = 1.0 / dt
    st["last_time"] = now

    ev = st.get("stop_event")
    if not st.get("running") or (ev is not None and ev.is_set()):
        vis = _draw_fps(driving_frame_rgb, st)
        return vis, st

    # ensure worker started
    st = ensure_worker(st)

    # enqueue frame (downscale if needed)
    frame = driving_frame_rgb
    if (frame.shape[1], frame.shape[0]) != (TARGET_W, TARGET_H):
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    if st["frame_buf"].full():
        try:
            st["frame_buf"].get_nowait()
        except queue.Empty:
            pass
    st["frame_buf"].put_nowait(frame)

    # ✅ if worker produced results, show them
    if st["result_buf"]:
        vis = st["result_buf"][-1]   # processed frame
    else:
        vis = driving_frame_rgb      # fallback: raw webcam

    vis = _draw_fps(vis, st)
    return vis, st

def live_prepare(src_path, st):
    token, msg = live_prepare_source(src_path)
    if token and token.get("ok"):
        st["src_ready"] = True
        st["first_frame"] = True
    else:
        st["src_ready"] = False
        st["first_frame"] = True
    return st, (msg or "")

def stop_worker(st):
    """Stop worker if running and clean up."""
    ev = st.get("stop_event")
    if ev and not ev.is_set():
        ev.set()
    t = st.get("worker_thread")
    if t and t.is_alive():
        t.join(timeout=2.0)
        
    st["worker_thread"] = None
    st["frame_buf"] = None
    st["result_buf"] = None
    st["first_frame"] = True
    st["worker_started"] = False
    st["running"] = False

def toggle_run(st):
    if not st.get("src_ready"):
        # always return expected outputs
        return st, "▶️ Start"  # unchanged label

    running = st.get("running", False)
    if running:
        # ---- STOP ----
        stop_worker(st)
        st["running"] = False
        # update button text back to Start
        return st, "▶️ Start"
    else:
        # ---- START ----
        st["running"] = True
        st = ensure_worker(st)
        # update button text to Stop
        return st, "⏹️ Stop"
    
def live_prepare_source(src_path):
    """
    Prepare source once per session (like run.py does before reading webcam).
    Returns a lightweight token (True/False) we keep in gr.State.
    """
    if not src_path:
        return None, "Please upload a source image first."
    ok = live_pipe.prepare_source(src_path, realtime=True)  # same flag as CLI realtime  # :contentReference[oaicite:6]{index=6}
    if not ok:
        return None, "No face detected in source. Try another image."
    # store copies so later we don't mutate live_pipe internals across sessions
    return {"ok": True}, "Source prepared ✅"


with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), js=js_func) as demo:
    gr.HTML(load_description("assets/gradio/gradio_title.md"))
    # ------------------- NEW: Live Webcam tab -------------------
    gr.Markdown("## 🔴 Live Generation (Webcam) ")
    gr.Markdown(load_description("assets/gradio/gradio_description_realtime.md"))
    with gr.Row():
        with gr.Column():
            live_src = gr.Image(label="Source Image (once)", type="filepath", height=TARGET_H, width=TARGET_W)
            live_prepare_btn = gr.Button("Prepare Source", variant="primary")
            live_status = gr.Markdown("")
        with gr.Column():
            live_cam = gr.Image(
                label="Driving (Webcam in Browser)",
                sources=["webcam"],  
                streaming=True,
                type="numpy",
                image_mode="RGB",
                height=TARGET_H,
                width=TARGET_W,
                elem_id="webcam"
            )
            gr.Markdown("**Note:** Webcam must be enabled in browser settings. If it doesn't work, try Chrome browser or check permissions.")
        with gr.Column():
            live_out = gr.Image(label="Output (Live)", type="numpy", format="jpeg", height=TARGET_H, width=TARGET_W)
            toggle_btn = gr.Button("▶️ Start", variant="primary")

    live_state = gr.State({"src_ready": False, "first_frame": True, "run": False, "last_vis": None})

    live_prepare_btn.click(live_prepare, inputs=[live_src, live_state], outputs=[live_state, live_status])
    
    live_cam.stream(live_step, inputs=[live_cam, live_state], outputs=[live_out, live_state], stream_every=0.1)  # 10fps
    
    toggle_btn.click(
        toggle_run,
        inputs=[live_state],
        outputs=[live_state, toggle_btn]
    )
    gr.Markdown("## ") 
    gr.Markdown("## 📷 Pre-Recorded Generation")
    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("🖼️ Source Image") as tab_image:
                    with gr.Accordion(open=True, label="Source Image"):
                        source_image_input = gr.Image(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s6.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s5.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s12.jpg")],
                            ],
                            inputs=[source_image_input],
                            cache_examples=False,
                        )

                with gr.TabItem("🎞️ Source Video") as tab_video:
                    with gr.Accordion(open=True, label="Source Video"):
                        source_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d9.mp4")],
                                [osp.join(example_video_dir, "d10.mp4")],
                                [osp.join(example_video_dir, "d11.mp4")],
                                [osp.join(example_video_dir, "d12.mp4")],
                                [osp.join(example_video_dir, "d13.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                            ],
                            inputs=[source_video_input],
                            cache_examples=False,
                        )

                tab_selection = gr.Textbox(value="Image", visible=False)  # default to Source Image

                tab_image.select(lambda: "Image", None, tab_selection)
                tab_video.select(lambda: "Video", None, tab_selection)

            with gr.Accordion(open=True, label="Cropping Options for Source Image or Video"):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("🎞️ Driving Video") as v_tab_video:
                    with gr.Accordion(open=True, label="Driving Video"):
                        driving_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d9.mp4")],
                                [osp.join(example_video_dir, "d10.mp4")],
                                [osp.join(example_video_dir, "d11.mp4")],
                                [osp.join(example_video_dir, "d12.mp4")],
                                [osp.join(example_video_dir, "d13.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                            ],
                            inputs=[driving_video_input],
                            cache_examples=False,
                        )
                with gr.TabItem("🖼️ Driving Image") as v_tab_image:
                    with gr.Accordion(open=True, label="Driving Image"):
                        driving_image_input = gr.Image(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s6.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s5.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s12.jpg")],
                            ],
                            inputs=[driving_image_input],
                            cache_examples=False,
                        )
                with gr.TabItem("📁 Driving Pickle") as v_tab_pickle:
                    with gr.Accordion(open=True, label="Driving Pickle"):
                        driving_pickle_input = gr.File(type="filepath", file_types=[".pkl"])
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d2.pkl")],
                                [osp.join(example_video_dir, "d8.pkl")],
                            ],
                            inputs=[driving_pickle_input],
                            cache_examples=False,
                        )

                v_tab_selection = gr.Textbox(value="Video", visible=False)
                v_tab_video.select(lambda: "Video", None, v_tab_selection)
                v_tab_image.select(lambda: "Image", None, v_tab_selection)
                v_tab_pickle.select(lambda: "Pickle", None, v_tab_selection)

                driving_text_input = gr.Textbox(value="", visible=False)  # placeholder so args align
                driving_audio_input = gr.Textbox(value="", visible=False) 

            gr.Markdown("**Tips:** Focus on the head, minimize shoulder movement, neutral expression in first frame.")
            with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                with gr.Row():
                    flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                    scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8,
                                                         maximum=3.2, step=0.05)
                    vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5,
                                                            maximum=0.5, step=0.01)
                    vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5,
                                                            maximum=0.5, step=0.01)

    with gr.Row():
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=False, label="relative motion")
                flag_stitching = gr.Checkbox(value=True, label="stitching")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0,
                                               step=0.02)
                cfg_scale = gr.Number(value=4.0, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                animation_region = gr.Radio(["exp", "pose", "lip", "eyes", "all"], value="all",
                                            label="animation region")
                flag_video_editing_head_rotation = gr.Checkbox(value=False, label="relative head rotation (v2v)")
                driving_smooth_observation_variance = gr.Number(value=1e-7, label="motion smooth strength (v2v)",
                                                                minimum=1e-11, maximum=1e-2, step=1e-8)
                flag_is_animal = gr.Checkbox(value=False, label="is_animal")

    gr.Markdown("---") 
    gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")

    with gr.Column():
        with gr.Row():
            with gr.Column():
                output_video_i2v = gr.Video(autoplay=False, label="The animated video in the original image space")
            with gr.Column():
                output_video_concat_i2v = gr.Video(autoplay=False, label="The animated video")
        with gr.Row():
            with gr.Column():
                output_image_i2i = gr.Image(format="png", type="numpy",
                                            label="The animated image in the original image space",
                                            visible=False)
            with gr.Column():
                output_image_concat_i2i = gr.Image(format="png", type="numpy", label="The animated image",
                                                   visible=False)
    with gr.Row():
        process_button_reset = gr.ClearButton(
            [source_image_input, source_video_input, driving_pickle_input, driving_video_input,
             driving_image_input, output_video_i2v, output_video_concat_i2v, output_image_i2i, output_image_concat_i2i],
            value="🧹 Clear")

    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            source_video_input,
            driving_video_input,
            driving_image_input,
            driving_pickle_input,
            driving_audio_input,
            driving_text_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            flag_video_editing_head_rotation,
            flag_is_animal,
            animation_region,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            driving_smooth_observation_variance,
            tab_selection,
            v_tab_selection,
            cfg_scale,

        ],
        outputs=[output_video_i2v, output_video_i2v, output_video_concat_i2v, output_video_concat_i2v,
                 output_image_i2i, output_image_i2i, output_image_concat_i2i, output_image_concat_i2i],
        show_progress=True
    )


if __name__ == '__main__':
    demo.launch(
        server_port=args.port,
        server_name=args.host_ip,
        share=False,
    )
