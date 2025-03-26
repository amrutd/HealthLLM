import streamlit as st
from transformers import pipeline
import time

# 1. APP CONFIGURATION ================================================
st.set_page_config(
    page_title="Counselor Guidance Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM STYLING ==================================================
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 200px;
        font-size: 16px;
    }
    .response-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4e79a7;
    }
    .stButton button {
        background-color: #4e79a7;
        color: white;
        transition: all 0.3s;
    }
    .stButton button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# 3. MODEL LOADING ===================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# 4. PROMPT ENGINEERING ==============================================
def generate_prompt(user_input, counseling_style):
    """Generate context-aware prompts for different counseling approaches"""
    style_prompts = {
        "CBT": (
            "As a CBT therapist, suggest techniques to address: '{input}'. "
            "Focus on identifying cognitive distortions and suggest behavioral experiments. "
            "Provide 2-3 concrete interventions."
        ),
        "Psychodynamic": (
            "From a psychodynamic perspective, analyze: '{input}'. "
            "Consider unconscious patterns and childhood influences. "
            "Suggest exploratory questions to reveal underlying conflicts."
        ),
        "Humanistic": (
            "Using humanistic approach, respond to: '{input}'. "
            "Focus on unconditional positive regard and self-actualization. "
            "Provide empathetic reflections and growth-oriented suggestions."
        ),
        "Solution-Focused": (
            "Using solution-focused therapy, address: '{input}'. "
            "Identify exceptions to the problem and small achievable steps. "
            "Suggest 2-3 scaling questions or miracle questions."
        )
    }
    return style_prompts[counseling_style].format(input=user_input)

# 5. MAIN APPLICATION ================================================
def main():
    st.title("🧠 Counselor Guidance Assistant")
    st.markdown("""
    *Professional support for mental health practitioners*  
    Enter a patient scenario below for evidence-based intervention suggestions.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Session Settings")
        counseling_style = st.selectbox(
            "Therapeutic Approach",
            ["CBT", "Psychodynamic", "Humanistic", "Solution-Focused"],
            index=0
        )
        creativity = st.slider("Response Creativity", 0.1, 1.0, 0.7)
        st.markdown("---")
        st.caption("""
        **Best Practices:**
        1. Describe specific behaviors/symptoms
        2. Include relevant history
        3. Note attempted interventions
        """)
    
    # Main input area
    case_description = st.text_area(
        "Describe the clinical challenge:",
        placeholder="E.g., My 28yo patient with social anxiety avoids all group situations despite previous exposure work...",
        height=250
    )
    
    # Example cases for quick testing
    with st.expander("💡 Example Cases"):
        examples = {
            "Depression": "45yo male with treatment-resistant depression, expresses hopelessness about ever improving",
            "Anxiety": "College student experiencing panic attacks before exams despite knowing the material well",
            "Relationship": "Couple stuck in pursue-withdraw pattern, escalating arguments about household responsibilities"
        }
        cols = st.columns(3)
        for i, (label, example) in enumerate(examples.items()):
            with cols[i]:
                if st.button(label):
                    case_description = example
    
    # Response generation
    if st.button("Get Clinical Suggestions", type="primary"):
        if not case_description.strip():
            st.warning("Please describe the clinical situation")
        else:
            with st.spinner("Generating evidence-based suggestions..."):
                try:
                    llm = load_model()
                    prompt = generate_prompt(case_description, counseling_style)
                    
                    # Simulate processing steps for better UX
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 20):
                        time.sleep(0.1)
                        progress_bar.progress(percent)
                    
                    response = llm(
                        prompt,
                        max_length=500,
                        do_sample=True,
                        temperature=creativity
                    )[0]['generated_text']
                    
                    # Display formatted response
                    st.markdown("## Clinical Recommendations")
                    with st.container():
                        st.markdown(f'<div class="response-card">{response}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Response tools
                        st.download_button(
                            "Save Recommendations",
                            data=response,
                            file_name=f"clinical_suggestions_{counseling_style}.txt"
                        )
                    
                except Exception as e:
                    st.error(f"Error generating suggestions: {str(e)}")

if __name__ == "__main__":
    main()


    

####################################### third


# import streamlit as st
# import joblib
# from transformers import pipeline
# import pandas as pd
# from PIL import Image
# import time

# # 1. PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# st.set_page_config(
#     page_title="Counselor Assistant Pro",
#     page_icon="🧠",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # 2. CUSTOM CSS (immediately after page config)
# st.markdown("""
# <style>
# :root {
#     --primary: #4e79a7;
#     --secondary: #f8f9fa;
# }

# .stApp {
#     max-width: 1000px;
#     margin: 0 auto;
# }

# .stTextArea textarea {
#     min-height: 200px;
#     background-color: var(--secondary);
#     border: 1px solid #e1e4e8 !important;
# }

# .stButton button {
#     background-color: var(--primary);
#     color: white;
#     transition: transform 0.2s;
# }

# .stButton button:hover {
#     transform: scale(1.02);
# }

# .response-bubble {
#     padding: 1.5rem;
#     background: var(--secondary);
#     border-radius: 0 15px 15px 15px;
#     position: relative;
#     margin: 1rem 0;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.1);
# }

# .tooltip {
#     position: relative;
#     display: inline-block;
# }

# .tooltip .tooltiptext {
#     visibility: hidden;
#     width: 120px;
#     background-color: #555;
#     color: #fff;
#     text-align: center;
#     border-radius: 6px;
#     padding: 5px;
#     position: absolute;
#     z-index: 1;
#     bottom: 125%;
#     left: 50%;
#     margin-left: -60px;
#     opacity: 0;
#     transition: opacity 0.3s;
# }

# .tooltip:hover .tooltiptext {
#     visibility: visible;
#     opacity: 1;
# }
# </style>
# """, unsafe_allow_html=True)

# # 3. MODEL LOADING
# @st.cache_resource(show_spinner=False)
# def load_models():
#     try:
#         clf = joblib.load("response_classifier.pkl")
#         vectorizer = joblib.load("tfidf_vectorizer.pkl")
#         llm = pipeline("text2text-generation", model="google/flan-t5-base")
#         return clf, vectorizer, llm
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.stop()

# # 4. HELPER FUNCTIONS


# # def build_prompt(patient_input, prediction, style="Balanced"):
# #     base_prompts = {
# #         "direct_advice": {
# #             "Balanced": "As a counselor, suggest 2-3 specific strategies to address this concern: '{input}'. Present them as options to explore together.",
# #             "More Empathetic": "First acknowledge the emotions in this statement: '{input}', then suggest one gentle strategy. Use supportive language.",
# #             "More Directive": "Provide 3 concrete action steps for this concern: '{input}'. Use clear, directive language."
# #         },
# #         "no_direct_advice": {
# #             "Balanced": "Respond to this client statement: '{input}'. Show empathy, reflect feelings, and ask one open-ended question.",
# #             "More Empathetic": "Create a deeply validating response to: '{input}'. Focus on emotional attunement and unconditional positive regard.",
# #             "More Directive": "For this statement: '{input}', reflect the content and gently guide toward insight with a clarifying question."
# #         }
# #     }
# #     template = base_prompts[prediction][style]
# #     return template.format(input=patient_input)


    


# def build_prompt(patient_input, prediction, style="Balanced"):
#     """Generate appropriate counseling prompts based on predicted response type"""
#     # Define all possible response types and styles
#     base_prompts = {
#         "direct_advice": {
#             "Balanced": "As a counselor, suggest 2-3 specific strategies to address: '{input}'. Present as options to explore together.",
#             "More Empathetic": "First acknowledge emotions in: '{input}', then suggest one gentle strategy using supportive language.",
#             "More Directive": "Provide 3 concrete action steps for: '{input}'. Use clear, directive language."
#         },
#         "no_direct_advice": {
#             "Balanced": "Respond to: '{input}'. Show empathy, reflect feelings, and ask one open-ended question.",
#             "More Empathetic": "Create a validating response to: '{input}'. Focus on emotional attunement.",
#             "More Directive": "For: '{input}', reflect content and guide toward insight with a clarifying question."
#         },
#         "reflection": {
#             "Balanced": "Respond to: '{input}'. Reflect content/emotions, then ask an exploratory question.",
#             "More Empathetic": "Create a reflective response to: '{input}' focusing on emotional validation.",
#             "More Directive": "For: '{input}', reflect content and guide toward deeper understanding."
#         },
#         "empathy": {
#             "Balanced": "Respond to: '{input}' with emotional validation and gentle exploration.",
#             "More Empathetic": "Create an empathetic response to: '{input}' focusing on connection.",
#             "More Directive": "For: '{input}', validate emotions while guiding conversation."
#         }
#     }
    
#     # Handle unknown prediction types
#     if prediction not in base_prompts:
#         st.warning(f"Unknown response type: {prediction}. Using default.")
#         prediction = "no_direct_advice"
    
#     # Handle unknown styles
#     if style not in base_prompts[prediction]:
#         st.warning(f"Unknown style: {style} for type {prediction}. Using Balanced.")
#         style = "Balanced"
    
#     template = base_prompts[prediction][style]
#     return template.format(input=patient_input)

# # 5. MAIN APP
# def main():
#     clf, vectorizer, llm = load_models()
    
#     # Sidebar controls
#     with st.sidebar:
#         st.title("Settings")
#         st.markdown("---")
#         model_temp = st.slider("Response Creativity", 0.1, 1.0, 0.7, 
#                             help="Higher values produce more creative responses")
#         response_style = st.selectbox(
#             "Preferred Response Style",
#             ["Balanced", "More Empathetic", "More Directive"],
#             index=0
#         )
#         st.markdown("---")
#         st.caption("""
#         **Instructions:**
#         1. Enter patient concern
#         2. Click 'Generate Response'
#         3. Review and refine as needed
#         """)

#     # Main content
#     st.title("🧠 Counselor Assistant Pro")
#     st.subheader("AI-powered support for mental health professionals")

#     # Split layout
#     col1, col2 = st.columns([3, 1])

#     with col1:
#         user_input = st.text_area(
#             "**Patient Concern**",
#             height=200,
#             placeholder="e.g., I've been feeling extremely anxious about my job and it's affecting my sleep...",
#             help="Describe the patient's current concerns or emotional state"
#         )

#     with col2:
#         st.markdown("**Common Concerns**")
#         example_buttons = {
#             "Anxiety": "I can't stop worrying about everything, it's exhausting.",
#             "Depression": "I've lost interest in all my hobbies and feel hopeless.",
#             "Relationship": "My partner and I keep having the same argument."
#         }
        
#         for label, example in example_buttons.items():
#             if st.button(label):
#                 user_input = example

#     # Response generation
#     if st.button("Generate Response", type="primary", use_container_width=True):
#         if not user_input.strip():
#             st.warning("Please enter a patient concern or select an example")
#         else:
#             with st.status("Analyzing concern...", expanded=True):
#                 st.write("🔍 Identifying key emotional themes...")
#                 time.sleep(0.5)
                
#                 progress_bar = st.progress(0)
#                 input_vector = vectorizer.transform([user_input])
#                 progress_bar.progress(30)
                
#                 prediction = clf.predict(input_vector)[0]
#                 progress_bar.progress(60)
                
#                 st.write(f"📊 Predicted approach: **{prediction.replace('_', ' ').title()}**")
#                 time.sleep(0.3)
#                 progress_bar.progress(90)
                
#                 st.write("💡 Crafting response...")
#                 prompt = build_prompt(user_input, prediction, response_style)
                
#                 try:
#                     response = llm(
#                         prompt,
#                         max_length=150,
#                         do_sample=True,
#                         temperature=model_temp
#                     )[0]['generated_text']
#                     progress_bar.progress(100)
#                     time.sleep(0.2)
                    
#                 except Exception as e:
#                     st.error(f"Error generating response: {e}")
#                     st.stop()
            
#             st.markdown("---")
#             st.subheader("Suggested Response")
            
#             with st.container(border=True):
#                 st.markdown(f"<div class='response-bubble'>{response.strip()}</div>", 
#                           unsafe_allow_html=True)
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.download_button("Save Response", response, file_name="counseling_response.txt")
#                 with col2:
#                     if st.button("Regenerate"):
#                         st.rerun()
#                 with col3:
#                     if st.button("Analyze Further"):
#                         st.session_state.analyze_mode = True
            
#             if st.session_state.get('analyze_mode', False):
#                 with st.expander("🔍 Response Analysis"):
#                     st.write("**Emotional Tone:**")
#                     st.write("**Therapeutic Techniques:**")

# if __name__ == "__main__":
#     main()




############## second  

# import streamlit as st
# import joblib
# from transformers import pipeline
# import pandas as pd
# from PIL import Image
# import time

# # Set page config - moved to top for Streamlit best practices
# st.set_page_config(
#     page_title="Counselor Assistant Pro",
#     page_icon="🧠",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# local_css("style.css")  # You can create this file for custom styles

# # Load ML models with better error handling
# @st.cache_resource(show_spinner=False)
# def load_models():
#     try:
#         clf = joblib.load("response_classifier.pkl")
#         vectorizer = joblib.load("tfidf_vectorizer.pkl")
#         llm = pipeline("text2text-generation", model="google/flan-t5-base")
#         return clf, vectorizer, llm
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.stop()

# clf, vectorizer, llm = load_models()

# # Sidebar for additional controls
# with st.sidebar:
#     st.title("Settings")
#     st.markdown("---")
#     model_temp = st.slider("Response Creativity", 0.1, 1.0, 0.7, 
#                           help="Higher values produce more creative responses")
#     response_style = st.selectbox(
#         "Preferred Response Style",
#         ["Balanced", "More Empathetic", "More Directive"],
#         index=0
#     )
#     st.markdown("---")
#     st.caption("""
#     **Instructions:**
#     1. Enter patient concern
#     2. Click 'Generate Response'
#     3. Review and refine as needed
#     """)

# # Main content area
# st.title("🧠 Counselor Assistant Pro")
# st.subheader("AI-powered support for mental health professionals")

# # Split layout into columns
# col1, col2 = st.columns([3, 1])

# with col1:
#     # Enhanced text input with placeholder example
#     user_input = st.text_area(
#         "**Patient Concern**",
#         height=200,
#         placeholder="e.g., I've been feeling extremely anxious about my job and it's affecting my sleep...",
#         help="Describe the patient's current concerns or emotional state"
#     )

# with col2:
#     st.markdown("**Common Concerns**")
#     example_buttons = {
#         "Anxiety": "I can't stop worrying about everything, it's exhausting.",
#         "Depression": "I've lost interest in all my hobbies and feel hopeless.",
#         "Relationship": "My partner and I keep having the same argument."
#     }
    
#     for label, example in example_buttons.items():
#         if st.button(label):
#             user_input = example

# # Response generation with enhanced UI
# if st.button("Generate Response", type="primary", use_container_width=True):
#     if not user_input.strip():
#         st.warning("Please enter a patient concern or select an example")
#     else:
#         # Create a status container
#         status = st.status("Analyzing concern...", expanded=True)
        
#         with status:
#             st.write("🔍 Identifying key emotional themes...")
#             time.sleep(0.5)
            
#             # Prediction with progress
#             progress_bar = st.progress(0)
#             input_vector = vectorizer.transform([user_input])
#             progress_bar.progress(30)
            
#             prediction = clf.predict(input_vector)[0]
#             progress_bar.progress(60)
            
#             st.write(f"📊 Predicted approach: **{prediction.replace('_', ' ').title()}**")
#             time.sleep(0.3)
#             progress_bar.progress(90)
            
#             # Generate response
#             st.write("💡 Crafting response...")
#             prompt = build_prompt(user_input, prediction, response_style)
            
#             try:
#                 response = llm(
#                     prompt,
#                     max_length=150,
#                     do_sample=True,
#                     temperature=model_temp
#                 )[0]['generated_text']
#                 progress_bar.progress(100)
#                 time.sleep(0.2)
                
#             except Exception as e:
#                 st.error(f"Error generating response: {e}")
#                 st.stop()
        
#         # Display response in a nice container
#         st.markdown("---")
#         st.subheader("Suggested Response")
        
#         with st.container(border=True):
#             st.markdown(f"<div class='response-box'>{response.strip()}</div>", 
#                        unsafe_allow_html=True)
            
#             # Response tools
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.download_button("Save Response", response, file_name="counseling_response.txt")
#             with col2:
#                 if st.button("Regenerate"):
#                     st.rerun()
#             with col3:
#                 if st.button("Analyze Further"):
#                     st.session_state.analyze_mode = True
        
#         # Optional analysis section
#         if st.session_state.get('analyze_mode', False):
#             with st.expander("🔍 Response Analysis"):
#                 st.write("**Emotional Tone:**")
#                 # Add emotion analysis here
                
#                 st.write("**Therapeutic Techniques:**")
#                 # Add technique analysis here

# # Helper functions
# def build_prompt(patient_input, prediction, style="Balanced"):
#     base_prompts = {
#         "direct_advice": {
#             "Balanced": "As a counselor, suggest 2-3 specific strategies to address this concern: '{input}'. Present them as options to explore together.",
#             "More Empathetic": "First acknowledge the emotions in this statement: '{input}', then suggest one gentle strategy. Use supportive language.",
#             "More Directive": "Provide 3 concrete action steps for this concern: '{input}'. Use clear, directive language."
#         },
#         "no_direct_advice": {
#             "Balanced": "Respond to this client statement: '{input}'. Show empathy, reflect feelings, and ask one open-ended question.",
#             "More Empathetic": "Create a deeply validating response to: '{input}'. Focus on emotional attunement and unconditional positive regard.",
#             "More Directive": "For this statement: '{input}', reflect the content and gently guide toward insight with a clarifying question."
#         }
#     }
    
#     template = base_prompts[prediction][style]
#     return template.format(input=patient_input)
# ##############














######## first 
# import streamlit as st
# import joblib
# from transformers import pipeline
# import pandas as pd



# # classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# # df["response_type"] = df["Response"].apply(lambda x: classifier(x)[0]["label"])

# st.set_page_config(page_title="Counselor Assistant (CPU-based)", layout="centered")


# # Load your ML model
# clf = joblib.load("response_classifier.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# # Set up the CPU-friendly model
# @st.cache_resource
# def load_llm():
#     return pipeline("text2text-generation", model="google/flan-t5-base")

# llm = load_llm()

# # Streamlit UI
# # st.set_page_config(page_title="Counselor Assistant (CPU-based)", layout="centered")
# st.title("🧠 Counselor Assistant (Free CPU Model)")
# st.subheader("Enter a patient concern to get a response suggestion")

# user_input = st.text_area("Enter the patient's concern:", height=200)

# def build_prompt(patient_input, prediction):
#     if prediction == "direct_advice":
#         return f"Suggest direct advice for a counselor to help with this concern: {patient_input}"
#     else:
#         return f"Give a reflective and empathetic response a counselor can say for this concern: {patient_input}"

# if st.button("Generate Suggestion"):
#     if user_input.strip() == "":
#         st.warning("Please enter the patient's concern.")
#     else:
#         input_vector = vectorizer.transform([user_input])
#         prediction = clf.predict(input_vector)[0]
#         st.markdown(f"**🧾 Predicted Response Style:** `{prediction}`")

#         with st.spinner("Thinking..."):
#             try:
#                 prompt = build_prompt(user_input, prediction)
#                 response = llm(prompt, max_length=100, do_sample=True)[0]['generated_text']
#                 st.markdown("### 💬 Suggested Counseling Response")
#                 st.write(response.strip())
#             except Exception as e:
#                 st.error(f"Error from local model: {e}")





