import gradio as gr
from wavelet import inference_wavelet
from denoise_model import inference_invdn
from restormer import inference_restormer
from uformer import inference_uformer
from nafnet import inference_nafnet

name = ["wavelet", "invdn", "restormer", "uformer", "nafnet"]
l = [inference_wavelet, inference_invdn, inference_restormer, inference_uformer, inference_nafnet]
inputs_images = [
    gr.components.Image(type="filepath", label="Input Image") for i in range(len(l))]
outputs_images = [
    gr.components.Image(type="numpy", label="Output Image") for i in range(len(l))
]

interface_images = [gr.Interface(
    fn=l[i],
    inputs=inputs_images[i],
    outputs=outputs_images[i],
    title="Image denoising",
    cache_examples=False,)
    for i in range(len(l))
]


app = gr.TabbedInterface(interface_list=interface_images,
                         tab_names = [str(i) for i in name])

app.launch()
