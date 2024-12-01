import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import tempfile
from dotenv import load_dotenv
import os
import re

class PDFProcessor:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
         
    def process_pdf(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        return text_splitter.split_documents(pages)

    def get_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        return FAISS.from_documents(chunks, embeddings)

    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini', openai_api_key=self.OPENAI_API_KEY)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            verbose=True
        )

    def generate_company_overview(self, conversation_chain):
        """Generate a brief company overview from the knowledge base"""
        overview_prompt = """
        Provide a concise 2-sentence overview of the company based on the uploaded documents. 
        Focus on the company's core business, main mission, or key distinguishing features.
        """
        
        response = conversation_chain({
            'question': overview_prompt,
            'chat_history': []
        })
        
        return response['answer']

class RAGApp:
    def __init__(self):
        self.processor = PDFProcessor()
        self.init_session_state()

    def init_session_state(self):
        # Initialize session state variables with defaults
        session_defaults = {
            'conversation': None,
            'chat_history': [],
            'processed_pdfs': False,
            'company_overview': None,
            'conversation_stage': 'initial',
            'email': None,
            'messages': []
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def validate_email(self, email):
        """Simple email validation"""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, email) is not None

    def mock_send_email(self, email):
        """Dummy email sending function"""
        print(f"Sending company profile to {email}")
        return True

    def add_message(self, role, message):
        """Add a message to the chat history"""
        st.session_state.messages.append({"role": role, "message": message})

    def display_chat_history(self):
        """Display the entire chat history"""
        st.sidebar.header("💬 Conversation History")
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.sidebar.write(f"👤 {msg['message']}")
            else:
                st.sidebar.write(f"🤖 {msg['message']}")

    def main(self):
        st.title("📚 Company Profile Chatbot")
        
        # Sidebar for navigation and chat history
        self.display_chat_history()

        # Navigation buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📄 Upload Documents"):
                st.session_state.conversation_stage = 'initial'
        with col2:
            if st.session_state.processed_pdfs and st.button("🏢 Company Overview"):
                st.session_state.conversation_stage = 'overview'
        with col3:
            if st.session_state.processed_pdfs and st.button("❓ Ask Questions"):
                st.session_state.conversation_stage = 'details'
        with col4:
            if st.session_state.processed_pdfs and st.button("📧 Get Profile"):
                st.session_state.conversation_stage = 'offer_profile'

        # PDF Upload Stage
        if st.session_state.conversation_stage == 'initial':
            st.header("📄 Upload Company Documents")
            pdf_docs = st.file_uploader("Upload your Company PDF Documents", type=['pdf'], accept_multiple_files=True)

            if st.button("Process PDFs") and pdf_docs:
                with st.spinner("Processing PDFs..."):
                    all_chunks = [chunk for pdf in pdf_docs for chunk in self.processor.process_pdf(pdf)]
                    vectorstore = self.processor.get_vectorstore(all_chunks)
                    st.session_state.conversation = self.processor.get_conversation_chain(vectorstore)
                    st.session_state.processed_pdfs = True
                    
                    # Generate company overview
                    st.session_state.company_overview = self.processor.generate_company_overview(st.session_state.conversation)
                    
                    # Add system message
                    self.add_message('system', "Company documents processed successfully!")
                    
                    st.session_state.conversation_stage = 'overview'
                    st.success("PDFs processed successfully!")

        # Company Overview Stage
        if st.session_state.processed_pdfs and st.session_state.conversation_stage == 'overview':
            st.header("🏢 Company Overview")
            st.write(st.session_state.company_overview)
            
            # Add overview to messages if not already added
            if not any(msg['message'] == st.session_state.company_overview for msg in st.session_state.messages):
                self.add_message('system', st.session_state.company_overview)
            
            want_more = st.radio("Would you like to know more about the company?", ('Yes', 'No'))
            
            if want_more == 'Yes':
                self.add_message('user', "Want to know more about the company")
                st.session_state.conversation_stage = 'details'

        # Details Exploration Stage
        if st.session_state.conversation_stage == 'details':
            st.header("❓ Ask Company Details")
            
            # Input for user question
            user_question = st.text_input("Ask a question about the company:")
            
            if user_question:
                # Add user message
                self.add_message('user', user_question)
                
                # Get response from conversation chain
                response = st.session_state.conversation({
                    'question': user_question,
                    'chat_history': st.session_state.chat_history
                })
                
                # Add bot response
                bot_answer = response['answer']
                self.add_message('system', bot_answer)
                
                # Display response
                st.write(f"🤖 Answer: {bot_answer}")
                
                # Update chat history
                st.session_state.chat_history.append((user_question, bot_answer))
            
            # Option to move to profile offer
            if st.button("I'd like the full company profile"):
                st.session_state.conversation_stage = 'offer_profile'

        # Profile Offer Stage
        if st.session_state.conversation_stage == 'offer_profile':
            st.header("📧 Company Profile Delivery")
            st.write("Would you like us to send the company profile to your email?")
            
            email_option = st.radio("Send Company Profile", ('Yes', 'No'))
            
            if email_option == 'Yes':
                self.add_message('user', "Interested in receiving company profile")
                st.session_state.conversation_stage = 'collect_email'
            else:
                self.add_message('system', "User declined company profile")
                st.write("Thank you for your interest!")

        # Email Collection Stage
        if st.session_state.conversation_stage == 'collect_email':
            st.header("📧 Email Collection")
            email = st.text_input("Please enter your email address:")
            
            if st.button("Confirm Email"):
                if self.validate_email(email):
                    # Mock email sending
                    if self.mock_send_email(email):
                        # Add messages
                        self.add_message('user', f"Email provided: {email}")
                        self.add_message('system', f"Company profile sent to {email}")
                        
                        st.success(f"Company profile sent to {email}!")
                        st.session_state.conversation_stage = 'complete'
                else:
                    st.error("Please enter a valid email address.")

        # Completion Stage
        if st.session_state.conversation_stage == 'complete':
            st.write("Thank you for your interest in our company!")
            self.add_message('system', "Conversation completed")

if __name__ == '__main__':
    app = RAGApp()
    app.main()