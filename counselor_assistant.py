

#################================================================================================================
#working 
#################================================================================================================


import streamlit as st
from transformers import pipeline
import time
from datetime import datetime

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
 :root {
  --primary_clr: #43a573;
  --secondary: #f8f9fa;
  --font_clr: #1a1a1a;
  --bg_clr: #e9f5ef;
  --bg_clr_g2: #d9ede3;
  --bg_clr_g3: #1b422e;
  --border_clr: #1b422e;
 }

 * {
  box-sizing: border-box;
 }

 html,body {
  font-family: sans-serif;
  line-height: 1.15;
  -webkit-text-size-adjust: 100%;
  -ms-text-size-adjust: 100%;
  -ms-overflow-style: scrollbar;
  -webkit-tap-highlight-color: transparent;
  margin:0;
  height: 100%;
  color: var(--font_clr);
  background-color: var(--bg_clr)
 }

 @-ms-viewport {
 width: device-width;
 }

 .stAppHeader, .stMain{
  background-color: var(--bg_clr)
 }

 .stSidebar{
  background-color: var(--bg_clr_g2);
 }

 .stTextArea textarea, .stTextArea > textarea {
  min-height: 250px;
  font-size: 16px;
  background-color: var(--bg_clr_g2);
  border-color: var(--border_clr);
  resize: none;
 }

 .responseCard {
  background-color: var(--bg_clr);
  border-radius: 10px;
  padding: 1.5rem;
  margin: 1rem 0;
  border-left: 4px solid var(--primary_clr);
 }
 
 .stButton button, .stButton > button {
  background-color: var(--primary_clr);
  color: white;
  transition: all 0.3s;
  border: 1px solid var(--border_clr);
 }
 
 .stButton button:hover, .stButton button:focus:not(:active) {
  opacity: 0.9;
  transform: translateY(-1px);
  color: white;
  background-color: var(--bg_clr_g3);
  border: 1px solid var(--border_clr);
 }

 /*
 .stSlider [role="slider"]{
  background-color: var(--bg_clr_g3);
 }

 .stSlider [role="slider"] .st-an, .stSlider [role="slider"] .st-ap, .stSlider [role="slider"] .st-aq,
 .stSlider [role="slider"] .st-ao, .stSlider [role="slider"] .st-cu, .stSlider [role="slider"] .st-cv,
 .stSlider [role="slider"] .st-am, .stSlider [role="slider"] .st-cw, .stSlider [role="slider"] .st-cx{
  background: linear-gradient(to right, rgb(27, 66, 46) 0%, 
  rgb(27, 66, 46) 72.2222%,
  rgba(151, 166, 195, 0.25) 72.2222%,
  rgba(151, 166, 195, 0.25) 100%) !important;
 }*/

 
 
 .progress-container {
  margin-top: -10px;
  margin-bottom: 15px;
 }
 
 .crisis-alert {
  background-color: #fff8f8;
  border-left: 5px solid #ff4b4b;
  padding: 1.5rem;
  margin: 1rem 0;
  border-radius: 0 8px 8px 0;
  box-shadow: 0 2px 8px rgba(255, 75, 75, 0.15);
  animation: pulse 2s infinite;
 }
 
 @keyframes pulse {
  0% { border-left-color: #ff4b4b; }
  50% { border-left-color: #ff9999; }
  100% { border-left-color: #ff4b4b; }
 }
 
 .crisis-alert strong {
  color: #d10000;
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

def get_references(approach):
    """Return evidence-based references with clinical guidelines"""
    references = {
        "CBT": {
            "text": "Beck, J. S. (2011). Cognitive Behavior Therapy: Basics and Beyond",
            "guide": "https://www.apa.org/pubs/books/cognitive-behavior-therapy"
        },
        "Psychodynamic": {
            "text": "McWilliams, N. (2020). Psychoanalytic Diagnosis",
            "guide": "https://www.guilford.com/books/Psychoanalytic-Diagnosis/McWilliams/9781462543694"
        },
        "Humanistic": {
            "text": "Rogers, C. (1951). Client-Centered Therapy",
            "guide": "https://www.nationalcounsellingsociety.org/about-therapy/types/humanistic"
        },
        "Solution-Focused": {
            "text": "De Shazer, S. (1988). Clues: Investigating Solutions in Brief Therapy",
            "guide": "https://www.solutionfocused.net/what-is-sfbt/"
        }
    }
    ref = references.get(approach, {
        "text": "Evidence-Based Practice in Psychology",
        "guide": "https://www.apa.org/practice/guidelines/evidence-based"
    })
    return f"{ref['text']} | [Clinical Guidelines]({ref['guide']})"

# 5. MAIN APPLICATION ================================================
def main():
    # Initialize session history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
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

        # Session history viewer
        if st.session_state.history:
            with st.expander("📚 Session History (Last 5)"):
                for i, session in enumerate(st.session_state.history[-5:][::-1]):
                    st.markdown(f"""
                    **Session {len(st.session_state.history)-i}** ({session['timestamp']})
                    - Approach: {session['approach']}
                    - Case: {session['case']}
                    """)
                    if st.button(f"View Details #{len(st.session_state.history)-i}", key=f"view_{i}"):
                        st.session_state.current_session = session
        
        
    
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
    
    # Crisis keywords detection
    CRISIS_KEYWORDS = ['suicide', 'self-harm', 'homicide', 'abuse', 'abused', 'kill myself', 'kill', 
    									'want to die', 'end my life', 'hurt myself', 'hurt someone','suicidal']
    
    # Response generation
    if st.button("Get Clinical Suggestions", type="primary"):
        if not case_description.strip():
            st.warning("Please describe the clinical situation")
        else:
            # Crisis detection
            if any(keyword in case_description.lower() for keyword in CRISIS_KEYWORDS):
                st.markdown("""
                <div class="crisis-alert">
                    <div style="font-size: 1.3rem;">⚠️ <strong>CRISIS ALERT</strong> - Immediate Action Required</div>
                    <br>
                    <div><strong>Clinical Protocols:</strong></div>
                    <ol>
                        <li>Assess immediate safety risk using direct questioning</li>
                        <li>Implement safety planning if risk is present</li>
                        <li>Do not leave patient alone if active suicidal/homicidal ideation exists</li>
                    </ol>
                    <div><strong>Emergency Resources:</strong></div>
                    <ul>
                        <li>🇺🇸 <strong>988 Suicide & Crisis Lifeline</strong> (24/7)</li>
                        <li>📱 <strong>Crisis Text Line</strong>: Text HOME to 741741</li>
                        <li>🌎 <strong>International Association for Suicide Prevention</strong>: <a href="https://www.iasp.info/resources/Crisis_Centres/">Find Local Help</a></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Clinician Guidance (Click for Protocol Details)"):
                    st.markdown("""
                    **Standard Crisis Response Protocol:**
                    1. **Direct Assessment**  
                       "Are you having thoughts of ending your life?"  
                       "Do you have a plan?"  
                       "Have you ever attempted before?"
                    
                    2. **Safety Planning**  
                       - Remove access to means  
                       - Identify support contacts  
                       - Create step-by-step coping strategies
                    
                    3. **Documentation**  
                       - Risk assessment findings  
                       - Actions taken  
                       - Follow-up plan
                    """)
                
                # Log crisis event
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'approach': "CRISIS INTERVENTION",
                    'case': "CRISIS DETECTED - " + case_description[:50] + "...",
                    'recommendations': "Session halted - emergency protocols activated"
                })
                st.stop()
            
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
                    
                    # Store session in history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'approach': counseling_style,
                        'case': case_description[:100] + "..." if len(case_description) > 100 else case_description,
                        'recommendations': response
                    })
                    
                    # Display formatted response
                    st.markdown("## Clinical Recommendations")
                    with st.container():
                        st.markdown(f'<div class="responseCard">{response}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Add references
                        st.markdown("---")
                        st.caption(f"**Reference:** {get_references(counseling_style)}")
                        
                        # Response tools
                        st.download_button(
                            "Save Recommendations",
                            data=f"Approach: {counseling_style}\n\n{response}",
                            file_name=f"clinical_suggestions_{counseling_style}.txt"
                        )
                    
                except Exception as e:
                    st.error(f"Error generating suggestions: {str(e)}")

if __name__ == "__main__":
    main()


##################================================================================================================
##################================================================================================================






##################================================================================================================
#first 
##################================================================================================================

# import streamlit as st
# from transformers import pipeline
# import time
# from datetime import datetime

# # 1. APP CONFIGURATION ================================================
# st.set_page_config(
#     page_title="Counselor Guidance Assistant",
#     page_icon="🧠",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # 2. CUSTOM STYLING ==================================================
# st.markdown("""
# <style>
#     .stTextArea textarea {
#         min-height: 200px;
#         font-size: 16px;
#     }
#     .response-card {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         border-left: 4px solid var(--primary_clr);
#     }
#     .stButton button {
#         background-color: var(--primary_clr);
#         color: white;
#         transition: all 0.3s;
#     }
#     .stButton button:hover {
#         opacity: 0.9;
#         transform: translateY(-1px);
#     }
#     .crisis-alert {
#         background-color: #ffdddd;
#         border-left: 5px solid #ff0000;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # 3. MODEL LOADING ===================================================
# @st.cache_resource(show_spinner=False)
# def load_model():
#     return pipeline("text2text-generation", model="google/flan-t5-base")

# # 4. PROMPT ENGINEERING ==============================================
# def generate_prompt(user_input, counseling_style):
#     """Generate context-aware prompts for different counseling approaches"""
#     style_prompts = {
#         "CBT": (
#             "As a CBT therapist, suggest techniques to address: '{input}'. "
#             "Focus on identifying cognitive distortions and suggest behavioral experiments. "
#             "Provide 2-3 concrete interventions."
#         ),
#         "Psychodynamic": (
#             "From a psychodynamic perspective, analyze: '{input}'. "
#             "Consider unconscious patterns and childhood influences. "
#             "Suggest exploratory questions to reveal underlying conflicts."
#         ),
#         "Humanistic": (
#             "Using humanistic approach, respond to: '{input}'. "
#             "Focus on unconditional positive regard and self-actualization. "
#             "Provide empathetic reflections and growth-oriented suggestions."
#         ),
#         "Solution-Focused": (
#             "Using solution-focused therapy, address: '{input}'. "
#             "Identify exceptions to the problem and small achievable steps. "
#             "Suggest 2-3 scaling questions or miracle questions."
#         )
#     }
#     return style_prompts[counseling_style].format(input=user_input)

# # def get_references(approach):
# #     """Return evidence-based references for each therapeutic approach"""
# #     references = {
# #         "CBT": "Beck, J. S. (2011). Cognitive Behavior Therapy: Basics and Beyond",
# #         "Psychodynamic": "McWilliams, N. (2020). Psychoanalytic Diagnosis",
# #         "Humanistic": "Rogers, C. (1951). Client-Centered Therapy",
# #         "Solution-Focused": "De Shazer, S. (1988). Clues: Investigating Solutions in Brief Therapy"
# #     }
# #     return references.get(approach, "Evidence-Based Practice in Psychology")


# def get_references(approach):
#     """Return evidence-based references with clinical guidelines"""
#     references = {
#         "CBT": {
#             "text": "Beck, J. S. (2011). Cognitive Behavior Therapy: Basics and Beyond",
#             "guide": "https://www.apa.org/pubs/books/cognitive-behavior-therapy"
#         },
#         "Psychodynamic": {
#             "text": "McWilliams, N. (2020). Psychoanalytic Diagnosis",
#             "guide": "https://www.guilford.com/books/Psychoanalytic-Diagnosis/McWilliams/9781462543694"
#         },
#         # Add other approaches similarly
#     }
#     ref = references.get(approach, {
#         "text": "Evidence-Based Practice in Psychology",
#         "guide": "https://www.apa.org/practice/guidelines/evidence-based"
#     })
#     return f"{ref['text']} | [Clinical Guidelines]({ref['guide']})"

# # 5. MAIN APPLICATION ================================================
# def main():
#     st.title("🧠 Counselor Guidance Assistant")
#     st.markdown("""
#     *Professional support for mental health practitioners*  
#     Enter a patient scenario below for evidence-based intervention suggestions.
#     """)
    
#     # Sidebar configuration
#     with st.sidebar:
#         st.title("Session Settings")
#         counseling_style = st.selectbox(
#             "Therapeutic Approach",
#             ["CBT", "Psychodynamic", "Humanistic", "Solution-Focused"],
#             index=0
#         )
#         creativity = st.slider("Response Creativity", 0.1, 1.0, 0.7)
#         st.markdown("---")
#         st.caption("""
#         **Best Practices:**
#         1. Describe specific behaviors/symptoms
#         2. Include relevant history
#         3. Note attempted interventions
#         """)
    
#     # Main input area
#     case_description = st.text_area(
#         "Describe the clinical challenge:",
#         placeholder="E.g., My 28yo patient with social anxiety avoids all group situations despite previous exposure work...",
#         height=250
#     )
    
#     # Example cases for quick testing
#     with st.expander("💡 Example Cases"):
#         examples = {
#             "Depression": "45yo male with treatment-resistant depression, expresses hopelessness about ever improving",
#             "Anxiety": "College student experiencing panic attacks before exams despite knowing the material well",
#             "Relationship": "Couple stuck in pursue-withdraw pattern, escalating arguments about household responsibilities"
#         }
#         cols = st.columns(3)
#         for i, (label, example) in enumerate(examples.items()):
#             with cols[i]:
#                 if st.button(label):
#                     case_description = example

# # Add this right before the "Response generation" section in your main() function
# CRISIS_KEYWORDS = ['suicide', 'self-harm', 'homicide', 'abuse', 'kill myself', 
# 'want to die', 'end my life', 'hurt myself', 'hurt someone']

# # Response generation
# if st.button("Get Clinical Suggestions", type="primary"):
# 	if not case_description.strip():
# 	    st.warning("Please describe the clinical situation")
# 	else:
# 	    # Crisis detection - add this block
# 	    if any(keyword in case_description.lower() for keyword in CRISIS_KEYWORDS):
# 	        st.error("""
# 	        ⚠️ CRISIS ALERT - Immediate action required
	        
# 	        **Emergency Protocols:**
# 	        1. Assess immediate safety risk
# 	        2. Contact emergency services if imminent danger exists
# 	        3. Do not leave the patient alone if risk is present
	        
# 	        **Emergency Resources:**
# 	        - National Suicide Prevention Lifeline: 988
# 	        - Crisis Text Line: Text HOME to 741741
# 	        - Local emergency services: 911
# 	        """)
# 	        st.stop()  # Halt further processing
	        
# 	    with st.spinner("Generating evidence-based suggestions..."):
# 	        # ... rest of your existing code ... 
    
#   	# Response generation
#   if st.button("Get Clinical Suggestions", type="primary"):
#       if not case_description.strip():
#           st.warning("Please describe the clinical situation")
#       else:
#           with st.spinner("Generating evidence-based suggestions..."):
#               try:
#                   llm = load_model()
#                   prompt = generate_prompt(case_description, counseling_style)
                  
#                   # Simulate processing steps for better UX
#                   progress_bar = st.progress(0)
#                   for percent in range(0, 101, 20):
#                       time.sleep(0.1)
#                       progress_bar.progress(percent)
                  
#                   response = llm(
#                       prompt,
#                       max_length=500,
#                       do_sample=True,
#                       temperature=creativity
#                   )[0]['generated_text']
                  
#                   # Display formatted response
#                   st.markdown("## Clinical Recommendations")
#                   with st.container():
#                       st.markdown(f'<div class="response-card">{response}</div>', 
#                        unsafe_allow_html=True)
                      
#                       # Add references
#                       st.markdown("---")
#                       st.caption(f"**Reference:** {get_references(counseling_style)}")
                      
#                       # Response tools
#                       st.download_button(
#                         "Save Recommendations",
#                         data=f"Approach: {counseling_style}\n\n{response}",
#                         file_name=f"clinical_suggestions_{counseling_style}.txt"
#                       )
                  
#               except Exception as e:
#                   st.error(f"Error generating suggestions: {str(e)}")

# if __name__ == "__main__":
#     main()


# #     if 'history' not in st.session_state:
# #     st.session_state.history = []

# # # Store each case and response
# # st.session_state.history.append({
# #     'case': case_description,
# #     'recommendations': response,
# #     'approach': counseling_style,
# #     'timestamp': datetime.now()
# # })

# # CRISIS_KEYWORDS = ['suicide', 'self-harm', 'homicide', 'abuse']

# # if any(keyword in case_description.lower() for keyword in CRISIS_KEYWORDS):
# #     st.warning("⚠️ Immediate action required - detected crisis language")
# #     st.markdown("""
# #     **Emergency Resources:**
# #     - National Suicide Prevention Lifeline: 988
# #     - Crisis Text Line: Text HOME to 741741
# #     """)

# #     def get_references(approach):
# #     references = {
# #         "CBT": "Beck, J. S. (2011). Cognitive Behavior Therapy: Basics and Beyond",
# #         "Psychodynamic": "McWilliams, N. (2020). Psychoanalytic Diagnosis",
# #         # ... other approaches
# #     }
# #     return references.get(approach, "Evidence-Based Practice in Psychology")