import { pipeline, env } from '@xenova/transformers';

// 1. فرض استخدام WebGPU لمعالجة المصفوفات وتجاوز المعالج المركزي (CPU)
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.allowLocalModels = true;

async function initializeVisionPipeline() {
    console.error("[System] Allocating WebGPU memory and caching model weights...");

    try {
        // 2. تحميل نموذج اكتشاف الكائنات وتوجيهه حصرياً إلى معالج الرسومات
        // بمجرد تحميله، سيتم تخزينه في IndexedDB للعمل دون إنترنت (Offline-first)
        const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            device: 'webgpu',
        });

        const imageUrl = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
        console.error("[System] Executing zero-latency inference on client hardware...");

        // 3. تمرير بيانات الإدخال وتطبيق عتبة دقة صارمة (Threshold)
        const output = await detector(imageUrl, { threshold: 0.85 });

        if (!output || output.length === 0) {
            console.error("[Inference] No objects detected matching confidence threshold.");
            return;
        }

        // 4. استخراج الإحداثيات والتصنيفات
        console.log(JSON.stringify(output, null, 2));

        /* 
           هيكلة المخرجات المتوقعة:
           [
             { "score": 0.98, "label": "couch", "box": { "xmin": 34, "ymin": 55, "xmax": 200, "ymax": 150 } }
           ]
        */

    } catch (error) {
        console.error("[Critical Failure] Pipeline crashed. Hardware acceleration may be disabled:", error);
    }
}

initializeVisionPipeline();