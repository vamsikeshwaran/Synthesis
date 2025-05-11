import streamlit as st
import subprocess
import os
import tempfile
import time

def run_in_environment(env_name, script_path, working_dir=None, *args):
    """Run a script in a conda environment with better error handling"""
    try:
        # Construct the command
        command = [
            "conda", "run", "-n", env_name, "python", script_path, *args
        ]
        
        # For debugging (will be shown only in the debug section if enabled)
        debug_info = f"Running command: {' '.join(command)} in {working_dir if working_dir else os.getcwd()}"
        
        # Execute the command in the specified working directory
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir
        )
        
        # Capture output for debugging
        stdout, stderr = process.communicate()
        
        # Wait for the process to complete
        if process.returncode != 0:
            return False, debug_info, stdout, stderr
        
        return True, debug_info, stdout, stderr
    except Exception as e:
        return False, f"Exception: {str(e)}", "", ""

def process_video(video_file, language, debug_container=None):
    """Process the video with DeepDub and Wav2Lip with detailed error reporting"""
    # Get paths (customize these to your environment)
    deepdub_dir = os.path.expanduser("~/Developer/myprojects/miniproject-deepdub")
    wav2lip_dir = os.path.expanduser("~/Developer/Wav2Lip")
    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints/wav2lip.pth")
    audio_output = os.path.join(deepdub_dir, "segments/SPEAKER_00_segment_0_refined_clonedvoice.wav")
    
    # Check if directories exist
    if not os.path.exists(deepdub_dir):
        if debug_container:
            debug_container.error(f"DeepDub directory not found: {deepdub_dir}")
        return False, None, f"DeepDub directory not found: {deepdub_dir}"
    
    if not os.path.exists(wav2lip_dir):
        if debug_container:
            debug_container.error(f"Wav2Lip directory not found: {wav2lip_dir}")
        return False, None, f"Wav2Lip directory not found: {wav2lip_dir}"
    
    if not os.path.exists(checkpoint_path):
        if debug_container:
            debug_container.error(f"Checkpoint file not found: {checkpoint_path}")
        return False, None, f"Checkpoint file not found: {checkpoint_path}"
    
    # First run DeepDub (voices.py)
    if debug_container:
        debug_container.info("Starting DeepDub processing...")
    
    success1, debug_info1, stdout1, stderr1 = run_in_environment(
        "deepdub", 
        "voices.py", 
        deepdub_dir, 
        "--video_file", 
        video_file, 
        "--lang", 
        language
    )
    
    if debug_container:
        debug_container.text(debug_info1)
        if not success1:
            debug_container.error("DeepDub processing failed")
            debug_container.text(f"StdOut: {stdout1}")
            debug_container.text(f"StdErr: {stderr1}")
    
    if not success1:
        return False, None, "DeepDub processing failed. Check the debug logs."
    
    # Check if audio file was created
    if not os.path.exists(audio_output):
        if debug_container:
            debug_container.error(f"Audio output file not created: {audio_output}")
        return False, None, f"Audio output file not created: {audio_output}"
    
    # Then run Wav2Lip (inference.py)
    if debug_container:
        debug_container.info("Starting Wav2Lip processing...")
    
    success2, debug_info2, stdout2, stderr2 = run_in_environment(
        "wav2lip", 
        "inference.py", 
        wav2lip_dir, 
        "--checkpoint_path", 
        checkpoint_path, 
        "--face", 
        video_file, 
        "--audio", 
        audio_output
    )
    
    if debug_container:
        debug_container.text(debug_info2)
        if not success2:
            debug_container.error("Wav2Lip processing failed")
            debug_container.text(f"StdOut: {stdout2}")
            debug_container.text(f"StdErr: {stderr2}")
    
    if not success2:
        return False, None, "Wav2Lip processing failed. Check the debug logs."
    
    # Try to find the output video (based on Wav2Lip's default output)
    results_dir = os.path.join(os.path.dirname(wav2lip_dir), "results")
    possible_outputs = [
        os.path.join(results_dir, "result_voice.mp4"),  # Standard output
        os.path.join(wav2lip_dir, "results", "result_voice.mp4"),  # In Wav2Lip dir
        os.path.join(os.path.dirname(video_file), "results", "result_voice.mp4")  # Near input
    ]
    
    output_video = None
    for path in possible_outputs:
        if os.path.exists(path):
            output_video = path
            break
    
    if output_video:
        return True, output_video, "Processing completed successfully."
    else:
        if debug_container:
            debug_container.error("Output video not found in expected locations")
            debug_container.text(f"Checked paths: {possible_outputs}")
        return False, None, "Output video not found in expected locations."

# Set up the Streamlit UI
st.title("Video Language Dubbing")
st.subheader("Convert videos to a different language")

# Add debug option
show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
debug_container = st.sidebar.container() if show_debug else None

# Environment check
if show_debug:
    debug_container.subheader("Environment Check")
    conda_installed = subprocess.run(["which", "conda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).returncode == 0
    debug_container.text(f"Conda installed: {conda_installed}")
    
    if conda_installed:
        # List conda environments
        envs = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        debug_container.text("Conda environments:")
        debug_container.code(envs.stdout)

# Language selection
language_options = ["Telugu", "Hindi", "English", "Tamil", "Kannada", "Malayalam"]
selected_language = st.selectbox("Select Target Language", language_options)

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Display a preview of the uploaded video
    st.video(uploaded_file)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    if show_debug:
        debug_container.text(f"Temporary file saved at: {video_path}")
    
    # Process button
    if st.button("Process Video"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Starting video processing...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        status_text.text("Processing with DeepDub and Wav2Lip...")
        progress_bar.progress(30)
        
        success, output_video, message = process_video(video_path, selected_language, debug_container)
            
        progress_bar.progress(100)
        
        if success and output_video:
            status_text.text("Video processed successfully!")
            st.success("Video processing completed!")
            
            # Display the output video
            st.subheader("Processed Video")
            st.video(output_video)
            
            # Provide download link
            with open(output_video, "rb") as file:
                st.download_button(
                    label="Download Dubbed Video",
                    data=file,
                    file_name=f"dubbed_video_{selected_language}.mp4",
                    mime="video/mp4"
                )
        else:
            status_text.text("Processing failed.")
            st.error(f"An error occurred: {message}")
            
            if show_debug:
                debug_container.error("Processing failed with detailed message:")
                debug_container.text(message)
                
            st.info("Suggestions for troubleshooting:")
            st.markdown("""
            1. Check if conda environments 'deepdub' and 'wav2lip' are correctly installed
            2. Verify that all paths in the code match your system setup
            3. Enable the 'Show Debug Information' option in the sidebar for more details
            4. Make sure the video file is in a compatible format
            """)
        
        # Clean up the temporary file
        try:
            os.unlink(video_path)
            if show_debug:
                debug_container.text(f"Temporary file removed: {video_path}")
        except Exception as e:
            if show_debug:
                debug_container.error(f"Error removing temporary file: {str(e)}")