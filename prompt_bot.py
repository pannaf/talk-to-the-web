from __future__ import annotations

import json
from typing import AsyncIterable
import fastapi_poe as fp
from modal import App, Image, asgi_app, exit
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup

fastapi_app = FastAPI()

# CORS middleware allows requests from any origin 
# this is needed so that it'll actually call the ask-question endpoint
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins, change this to a specific list for production
    allow_credentials=True,
    allow_methods=["*"],  # allows all HTTP methods
    allow_headers=["*"],  # allows all headers
)

def get_question_response(question, context):
    """Get a response to the question using the provided context."""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [{"role": "system", "content": f"Context: {context}"},
                     {"role": "user", "content": f"Question: {question}"}]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer <YOUR_API_KEY>" # TODO: you need to include your api key here
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()

@fastapi_app.post("/ask-question/")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    context = data.get("context")

    response_data = get_question_response(question, context)

    return {"response": response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response received.")}

class InteractiveWebsiteBot(fp.PoeBot):
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:

        url = request.query[-1].content.strip()
        if not url.startswith("http"):
            yield fp.PartialResponse(text="Please provide a valid URL.")
            return

        response = requests.get(url)

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup.find_all('script'):
            script.decompose()

        for a_tag in soup.find_all('a'):
            a_tag.unwrap()

        cleaned_html = str(soup.prettify())

        cleaned_lines = []
        for line in cleaned_html.splitlines():
            if "http" not in line:
                cleaned_lines.append(line)

        final_html = "\n".join(cleaned_lines)

        prompt = f"""
        Put the following HTML in an interactive HTML page that lets the user ask questions about the contents. The HTML should allow users to click on any text, and when they do, that text should appear highlighted in yellow. They should be able to ask a question about it in a text box below the web content. A button should be able to be pressed to submit the question. The selected text should be appended to the 'question' and sent as 'question'. The full website content should be sent as 'context' to an API endpoint at '/ask-question/'. The response from the API should be displayed on the page. Please generate the HTML and JavaScript code needed for this. The website should look professional and be user-friendly. Be sure to use colors that are easy to read and navigate.:

        The HTML is:

        {final_html}

        Include all the content in the HTML above. And ensure a user can ask a question about any part of the content. The format of your output HTML should be:
        - Website content, pulled from the website content above. Focus on the main content area. Leave out any table of contents or anything extraneous.
        - A text box for the user to ask a question.
        - A button to submit the question.

        The interaction should be such that a user can highlight a portion of the website content and ask a question about it. The question should be sent to an API endpoint that will return a response. The response should be displayed on the page.

        The API endpoint 'https://pannaf--interactive-website-bot-model-fastapi-app-dev.modal.run/custom/ask-question/' accepts POST requests with JSON data in the following format:
        {{
            "context": "The web page content",
            "question": "The user's question"
        }}
        The format of the response is:
        {{
            "response": "The response to the user's question"
        }}
        """

        request.query = [fp.ProtocolMessage(role="system", content=prompt)]

        async for msg in fp.stream_request(request, "Gemini-1.5-Pro-2M", request.access_key):
            yield msg

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(server_bot_dependencies={"Gemini-1.5-Pro-2M": 1})


REQUIREMENTS = ["fastapi-poe==0.0.47", "requests", "beautifulsoup4"]
image = Image.debian_slim().pip_install(*REQUIREMENTS)
app = App(name="interactive-website-bot", image=image)

@app.cls()
class Model:
    access_key: str = "<YOUR_ACCESS_KEY>" # TODO you need to include your access key here
    bot_name: str = "DrVicFrankenstein" # TODO you need to include your bot name here.. mine was this :)

    @exit()
    def sync_settings(self):
        """Syncs bot settings on server shutdown."""
        if self.bot_name and self.access_key:
            try:
                fp.sync_bot_settings(self.bot_name, self.access_key)
            except Exception:
                print("\n*********** Warning ***********")
                print(
                    "Bot settings sync failed. For more information, see: https://creator.poe.com/docs/server-bots-functional-guides#updating-bot-settings"
                )
                print("\n*********** Warning ***********")

    @asgi_app()
    def fastapi_app(self):
        bot = InteractiveWebsiteBot()
        if not self.access_key:
            print(
                "Warning: Running without an access key. Please remember to set it before production."
            )
            app = fp.make_app(bot, allow_without_key=True)
        else:
            app = fp.make_app(bot, access_key=self.access_key)

        app.mount("/custom", fastapi_app)
        return app

@app.local_entrypoint()
def main():
    Model().run.remote()

