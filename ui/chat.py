import streamlit as st
# from ..Classes.inference import make_pipeline, MultimodalSignLM
import time 
# import ollama
import tempfile
import cv2
import numpy as np

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def stream_chat(model, messages, temperature=0.7):
    try:
        """
        stream = ollama.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
        """
        print(f"Model: {model}")
        for message in range(20):
            yield f"Message {message} from the model. "
            time.sleep(1)
    except Exception as e:
        print(f"Error in stream_chat: {str(e)}")

def main():
    st.title("Multimodal Sign LM")
    
    use_camera = st.sidebar.button("Iniciar cámara")
    if use_camera:
        btn_stop = st.sidebar.button("Detener", key="stop")

    stframe = st.empty()

    model = st.sidebar.selectbox("Choose a model", ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.2-1B", "Llama-3.2-3B"])

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe un mensaje:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()

                with st.spinner("generando..."):
                    try:
                        message = st.write_stream(stream_chat(model, st.session_state["messages"]))
                        st.session_state["messages"].append({"role": "assistant", "content": message})

                        duration = time.time() - start_time
                        st.write(f"Duración: {duration:.2f}s")

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        print(f"Error: {str(e)}")

    if use_camera:
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            st.error("No se pudo acceder a la cámara")
            return
        
        while True:
            ret, frame = vid.read()
            if not ret:
                st.warning("No se pudo leer el frame")
                break

            # Convertir a RGB para Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")
            
            # Control de velocidad de actualización (10 FPS aprox.)
            time.sleep(0.1)
            
            # Permitir salir con una condición de interrupción (por ejemplo, un botón)
            if btn_stop:
                break
        
        vid.release()
        st.sidebar.text("Cámara detenida.")

if __name__ == "__main__":
    main()