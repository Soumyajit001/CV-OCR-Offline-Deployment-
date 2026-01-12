"""
Streamlit Visualization App
============================

A simple single-page Streamlit app to visualize outputs from:
- Part A: Human & Animal Detection
- Part B: Industrial OCR
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path to import main modules
sys.path.append(str(Path(__file__).parent))

from main import ObjectDetector, HumanAnimalClassifier, VideoProcessor, IndustrialOCR


def save_detection_results(image_name: str, detections: list, annotated_image: np.ndarray, 
                          summary: dict, output_dir: Path = Path("./outputs")):
    """
    Save detection results, annotated image, and summary to output directory.
    
    Args:
        image_name: Original image filename
        detections: List of detection results
        annotated_image: Annotated image array
        summary: Summary dictionary with statistics
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_name).stem
    
    # Save annotated image
    annotated_image_path = output_dir / f"{base_name}_annotated_{timestamp}.jpg"
    cv2.imwrite(str(annotated_image_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    # Save detailed detection results
    results_data = {
        "image_name": image_name,
        "timestamp": timestamp,
        "total_detections": len(detections),
        "detections": detections
    }
    results_path = output_dir / f"{base_name}_detections_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary
    summary_data = {
        "image_name": image_name,
        "timestamp": timestamp,
        "summary": summary,
        "annotated_image_path": str(annotated_image_path),
        "results_path": str(results_path)
    }
    summary_path = output_dir / f"{base_name}_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return {
        "annotated_image": annotated_image_path,
        "results": results_path,
        "summary": summary_path
    }

st.set_page_config(
    page_title="AI Vision System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Vision System: Detection & OCR")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Application",
    ["Human & Animal Detection", "Industrial OCR"]
)

if app_mode == "Human & Animal Detection":
    st.header("Part A: Human & Animal Detection")
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    detector_model_path = st.sidebar.text_input(
        "Detector Model Path (optional)",
        value="",
        help="Leave empty to use pretrained model"
    )
    classifier_model_path = st.sidebar.text_input(
        "Classifier Model Path (optional)",
        value="",
        help="Leave empty to use pretrained model"
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Input method
    input_method = st.radio(
        "Input Method",
        ["Upload Image", "Process Video", "View Output Videos"],
        horizontal=True
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect humans and animals"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                # Initialize models
                try:
                    with st.spinner("Loading models..."):
                        detector = ObjectDetector(
                            model_path=detector_model_path if detector_model_path else None
                        )
                        classifier = HumanAnimalClassifier(
                            model_path=classifier_model_path if classifier_model_path else None
                        )
                    
                    # Run detection
                    with st.spinner("Running detection..."):
                        detections = detector.detect(img_array, confidence_threshold)
                        
                        # Refine with classifier
                        annotated_image = img_array.copy()
                        results = []
                        
                        for det in detections:
                            x1, y1, x2, y2 = map(int, det['bbox'])
                            
                            # Crop and classify
                            crop = img_array[y1:y2, x1:x2]
                            if crop.size > 0:
                                class_name, conf = classifier.classify(crop)
                                det['class_name'] = class_name
                                det['confidence'] = conf
                                
                                # Draw on image
                                color = (0, 255, 0) if class_name == 'human' else (255, 0, 0)
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                                label = f"{class_name}: {conf:.2f}"
                                cv2.putText(annotated_image, label, (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                results.append({
                                    "class": class_name,
                                    "confidence": float(conf),
                                    "bbox": {
                                        "x1": int(x1),
                                        "y1": int(y1),
                                        "x2": int(x2),
                                        "y2": int(y2),
                                        "width": int(x2 - x1),
                                        "height": int(y2 - y1)
                                    },
                                    "detection_confidence": float(det.get('confidence', 0.0))
                                })
                        
                        st.image(annotated_image, use_container_width=True)
                        
                        # Display results
                        if results:
                            st.subheader("Detection Summary")
                            st.json(results)
                            st.metric("Total Detections", len(results))
                            humans = sum(1 for r in results if r['class'] == 'human')
                            animals = sum(1 for r in results if r['class'] == 'animal')
                            st.metric("Humans", humans)
                            st.metric("Animals", animals)
                            
                            # Prepare summary
                            summary = {
                                "total_detections": len(results),
                                "humans": humans,
                                "animals": animals,
                                "confidence_threshold": confidence_threshold
                            }
                            
                            # Save results to output folder
                            try:
                                saved_files = save_detection_results(
                                    uploaded_file.name,
                                    results,
                                    annotated_image,
                                    summary
                                )
                                st.success("✅ Results saved successfully!")
                                
                                # Display saved files in expander
                                with st.expander("📁 View Saved Files", expanded=False):
                                    st.write("**Saved files:**")
                                    st.write(f"- **Annotated Image:** `{saved_files['annotated_image']}`")
                                    st.write(f"- **Detection Results (JSON):** `{saved_files['results']}`")
                                    st.write(f"- **Summary (JSON):** `{saved_files['summary']}`")
                                    st.write(f"\n**Location:** `./outputs/`")
                                    
                                    # Show download buttons
                                    st.write("\n**Download files:**")
                                    col_d1, col_d2, col_d3 = st.columns(3)
                                    
                                    with col_d1:
                                        with open(saved_files['annotated_image'], 'rb') as f:
                                            st.download_button(
                                                "📷 Annotated Image",
                                                f.read(),
                                                file_name=saved_files['annotated_image'].name,
                                                mime="image/jpeg"
                                            )
                                    with col_d2:
                                        with open(saved_files['results'], 'r') as f:
                                            st.download_button(
                                                "📊 Detection Results",
                                                f.read(),
                                                file_name=saved_files['results'].name,
                                                mime="application/json"
                                            )
                                    with col_d3:
                                        with open(saved_files['summary'], 'r') as f:
                                            st.download_button(
                                                "📋 Summary",
                                                f.read(),
                                                file_name=saved_files['summary'].name,
                                                mime="application/json"
                                            )
                            except Exception as save_error:
                                st.warning(f"Could not save results: {str(save_error)}")
                        else:
                            st.info("No objects detected.")
                            
                            # Save empty results for record keeping
                            try:
                                summary = {
                                    "total_detections": 0,
                                    "humans": 0,
                                    "animals": 0,
                                    "confidence_threshold": confidence_threshold
                                }
                                saved_files = save_detection_results(
                                    uploaded_file.name,
                                    [],
                                    annotated_image,
                                    summary
                                )
                                st.info(f"Results saved (no detections) to: `{saved_files['summary'].name}`")
                            except Exception as save_error:
                                pass
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Note: Models may need to be trained first. Check the training scripts.")
    
    elif input_method == "Process Video":
        st.subheader("Video Processing")
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to process"
        )
        
        if uploaded_video is not None:
            # Save uploaded video
            video_path = Path("./test_videos") / uploaded_video.name
            video_path.parent.mkdir(exist_ok=True)
            
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            st.success(f"Video saved to {video_path}")
            
            # Process video
            if st.button("Process Video"):
                try:
                    with st.spinner("Processing video..."):
                        detector = ObjectDetector(
                            model_path=detector_model_path if detector_model_path else None
                        )
                        classifier = HumanAnimalClassifier(
                            model_path=classifier_model_path if classifier_model_path else None
                        )
                        processor = VideoProcessor(detector, classifier)
                        
                        output_path = Path("./outputs") / f"annotated_{uploaded_video.name}"
                        output_path.parent.mkdir(exist_ok=True)
                        
                        processor.process_video(str(video_path), str(output_path))
                        
                        # Create video processing summary
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = Path(uploaded_video.name).stem
                        summary_path = Path("./outputs") / f"{base_name}_video_summary_{timestamp}.json"
                        
                        # Get video info
                        cap = cv2.VideoCapture(str(output_path))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        video_summary = {
                            "input_video": uploaded_video.name,
                            "output_video": output_path.name,
                            "timestamp": timestamp,
                            "video_info": {
                                "frames": frame_count,
                                "fps": fps,
                                "resolution": f"{width}x{height}",
                                "duration_seconds": frame_count / fps if fps > 0 else 0
                            },
                            "processing_settings": {
                                "confidence_threshold": confidence_threshold,
                                "detector_model": detector_model_path or "pretrained",
                                "classifier_model": classifier_model_path or "pretrained"
                            }
                        }
                        
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(summary_path, 'w') as f:
                            json.dump(video_summary, f, indent=2)
                        
                        st.success(f"Video processed! Output saved to {output_path}")
                        st.info(f"📄 Video summary saved to: `{summary_path.name}`")
                        
                        # Show output video
                        st.video(str(output_path))
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:  # View Output Videos
        st.subheader("Output Videos")
        output_dir = Path("./outputs")
        
        if output_dir.exists():
            video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))
            
            if video_files:
                for video_path in video_files:
                    st.video(str(video_path))
                    st.caption(f"File: {video_path.name}")
            else:
                st.info("No output videos found. Process some videos first!")
        else:
            st.info("Output directory does not exist yet.")

elif app_mode == "Industrial OCR":
    st.header("Part B: Industrial OCR")
    
    st.sidebar.subheader("OCR Configuration")
    use_preprocessing = st.sidebar.checkbox(
        "Enable Image Preprocessing",
        value=True,
        help="Apply contrast enhancement and denoising"
    )
    
    # Input method
    input_method = st.radio(
        "Input Method",
        ["Upload Image", "Process Directory", "View OCR Results"],
        horizontal=True
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an industrial image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image with stenciled/painted text"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("OCR Results")
                
                try:
                    with st.spinner("Running OCR..."):
                        ocr = IndustrialOCR()
                        
                        # Save temporary image
                        temp_path = Path("./temp_ocr_image.jpg")
                        image.save(temp_path)
                        
                        # Extract text
                        result = ocr.extract_text(str(temp_path))
                        
                        # Clean up
                        temp_path.unlink()
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Display extracted text
                            st.subheader("Extracted Text")
                            st.text_area(
                                "Text",
                                value=result.get("text", ""),
                                height=200,
                                disabled=True
                            )
                            
                            # Display confidence
                            if "average_confidence" in result:
                                st.metric(
                                    "Average Confidence",
                                    f"{result['average_confidence']:.2%}"
                                )
                            
                            # Display structured results
                            if "structured_text" in result:
                                st.subheader("Structured Results")
                                st.json(result["structured_text"])
                            
                            # Show preprocessing if enabled
                            if use_preprocessing:
                                st.subheader("Preprocessed Image")
                                processed = ocr.preprocess_image(img_array)
                                st.image(processed, use_container_width=True, clamp=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Note: PaddleOCR may need to be installed. Check requirements.txt.")
    
    elif input_method == "Process Directory":
        st.subheader("Batch Processing")
        input_dir = st.text_input(
            "Input Directory",
            value="./test_images",
            help="Directory containing images to process"
        )
        output_dir = st.text_input(
            "Output Directory",
            value="./outputs/ocr_results",
            help="Directory to save OCR results"
        )
        
        if st.button("Process Directory"):
            try:
                with st.spinner("Processing images..."):
                    ocr = IndustrialOCR()
                    results = ocr.process_directory(input_dir, output_dir)
                    
                    st.success(f"Processed {len(results)} images!")
                    st.json(results)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # View OCR Results
        st.subheader("OCR Results")
        results_dir = Path("./outputs/ocr_results")
        
        if results_dir.exists():
            json_files = list(results_dir.glob("*_ocr_result.json"))
            
            if json_files:
                selected_file = st.selectbox(
                    "Select result file",
                    [f.name for f in json_files]
                )
                
                if selected_file:
                    with open(results_dir / selected_file, 'r') as f:
                        result = json.load(f)
                    
                    st.json(result)
            else:
                st.info("No OCR results found. Process some images first!")
        else:
            st.info("Results directory does not exist yet.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        AI Vision System - Offline Detection & OCR
    </div>
    """,
    unsafe_allow_html=True
)
