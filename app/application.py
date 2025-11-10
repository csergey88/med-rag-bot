from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from app.components.retrivier import create_retrieval_qa_chain
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH,CHUNK_SIZE,CHUNK_OVERLAP

import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

logger=get_logger(__name__)

app=Flask(__name__)
app.secret_key=os.urandom(24)

from markupsafe import Markup

def nl2br(s):
    return Markup(s.replace("\n", "<br>"))

app.jinja_env.filters["nl2br"] = nl2br


@app.route("/",methods=["GET","POST"])
def index():
    try:
        if "messages" not in session:
            session["messages"]=[]
        
        if request.method=="POST":
            prompt=request.form["prompt"]
            
            messages=session["messages"]
            messages.append({"role":"user","content":prompt})
            session["messages"]=messages
            
            retrieval_qa_chain=create_retrieval_qa_chain()
            if not retrieval_qa_chain:
                raise CustomException("Retrieval QA chain not found")

            # Invoke retrieval QA chain
            logger.info(f"Invoking retrieval QA chain with prompt: {prompt}")
            response=retrieval_qa_chain.invoke({"query":prompt})
            logger.info(f"Response received: {response}")
            
            # Extract result from response
            if isinstance(response, dict):
                result=response.get("result") or response.get("answer") or str(response)
            else:
                result=str(response)
            
            logger.info(f"Result extracted: {result}")
            
            # Add assistant response to messages
            if result and result != "No answer":
                messages.append({"role":"assistant","content":result})
            else:
                messages.append({"role":"assistant","content":"I couldn't generate a response. Please try again."})
            session["messages"]=messages

            return redirect(url_for("index"))
            
    except Exception as e:
        import traceback
        error_msg = f"Error in index: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template("index.html", messages=session.get("messages",[]), error=str(e))

    return render_template("index.html",messages=session.get("messages",[]))

@app.route("/clear",methods=["GET"])
def clear():
    session["messages"]=[]
    return redirect(url_for("index"))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001,debug=True,use_reloader=False)
