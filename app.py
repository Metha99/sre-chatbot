import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config first, this must be the first command in the script
st.set_page_config(page_title="SRE Error Helper", page_icon="🔧", layout="centered")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("errors.csv")
    df["embedding"] = df["Error Message"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# Custom CSS for the chat bubble and avatar
st.markdown("""
    <style>
        /* Overall container for the chat window */
        #chat-container {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 350px;
            height: 400px;
            background-color: #2b2b2b;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            z-index: 9999;
        }

        /* Header for the chat window */
        #chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #fff;
            font-size: 16px;
            font-weight: bold;
        }

        /* Avatar */
        #chat-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-image: url('https://i.pinimg.com/originals/4f/ab/88/4fab88d0a3e8b7b3ff72b5be6c000106.gif');
            background-size: cover;
        }

        /* Chat messages container */
        #chat-messages {
            height: 250px;
            overflow-y: auto;
            margin-top: 10px;
            color: white;
            font-size: 14px;
        }

        /* Chat input container */
        #chat-input-container {
            display: flex;
            margin-top: 10px;
            justify-content: space-between;
        }

        #chat-input {
            width: 80%;
            padding: 10px;
            border-radius: 10px;
            border: none;
            font-size: 14px;
        }

        #chat-send {
            width: 15%;
            background-color: #4CAF50;
            border: none;
            color: white;
            border-radius: 10px;
            cursor: pointer;
        }

        #chat-send:hover {
            background-color: #45a049;
        }

        /* Close button for the chat window */
        .close-chat {
            font-size: 20px;
            color: #fff;
            cursor: pointer;
        }

        /* Glow effect for the input box */
        .glowing-input {
            animation: glowing 1500ms ease-in-out infinite;
            border: 1px solid #4CAF50;
        }

        /* Keyframes for glowing effect */
        @keyframes glowing {
            0% { border-color: #4CAF50; }
            50% { border-color: #00FF00; }
            100% { border-color: #4CAF50; }
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("🔧 Ask Niel")

# Display the chat window with avatar
st.markdown("""
    <div id="chat-container">
        <div id="chat-header">
            <div id="chat-avatar"></div>
            Ask Niel
            <div class="close-chat" onclick="document.getElementById('chat-container').style.display='none';">❌</div>
        </div>
        <div id="chat-messages"></div>
        <div id="chat-input-container">
            <input type="text" id="chat-input" class="glowing-input" placeholder="Ask a question..." />
            <button id="chat-send" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        // Make the chat window draggable
        let chatWindow = document.getElementById("chat-container");
        chatWindow.onmousedown = function(event) {
            let shiftX = event.clientX - chatWindow.getBoundingClientRect().left;
            let shiftY = event.clientY - chatWindow.getBoundingClientRect().top;

            document.onmousemove = function(event) {
                chatWindow.style.left = event.clientX - shiftX + 'px';
                chatWindow.style.top = event.clientY - shiftY + 'px';
            };

            chatWindow.onmouseup = function() {
                document.onmousemove = null;
                chatWindow.onmouseup = null;
            };
        };

        function sendMessage() {
            var inputBox = document.getElementById("chat-input");
            var query = inputBox.value;

            if (query !== "") {
                document.getElementById("chat-messages").innerHTML += "<p><strong>You:</strong> " + query + "</p>";

                // Simulate response
                setTimeout(function() {
                    document.getElementById("chat-messages").innerHTML += "<p><strong>Ask Niel:</strong> " + query + " response!</p>";
                }, 1000);
            }
            inputBox.value = "";
        }
    </script>
""", unsafe_allow_html=True)

# Input field for user queries
query = st.text_input("Enter the error you're seeing:")

# Process the query if user enters something
if query:
    # Find the best match from the dataframe
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in df["embedding"]]
    best_idx = scores.index(max(scores))
    
    # Show the best match found
    st.subheader("🔍 Best Match Found")
    st.write(f"**Error Code:** {df.iloc[best_idx]['Error Code']}")
    st.write(f"**Error Message:** {df.iloc[best_idx]['Error Message']}")
    st.write(f"**Likely Cause:** {df.iloc[best_idx]['Cause']}")
    st.write(f"**Suggested Fix:** {df.iloc[best_idx]['Resolution Steps']}")
