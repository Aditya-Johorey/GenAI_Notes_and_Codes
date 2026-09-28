To use your Hugging Face Fine-grained token inside the HTTP Request node in n8n, follow this configuration exactly to authenticate your request and properly capture the generated image file. [1, 2] 
## 1. Authentication Setup

   1. In the HTTP Request node, click on the Authentication dropdown and select Generic Credential Type. [3] 
   2. For Generic Auth Type, choose Header Auth. [4] 
   3. Click on the Credential for Header Auth dropdown and select Create New Credential. [3] 
   4. Configure the credential fields exactly like this:
   * Name: Authorization
      * Value: Bearer hf_your_actual_token_here (Make sure to leave a space between Bearer and your token string) [2, 4] 
   5. Click Save and go back to the node parameters. [5] 

------------------------------
## 2. Node Parameters & Endpoint Configuration
Configure the core parameters of your HTTP Request node to route your payload to the FLUX.1-schnell model:

* 
* Method: POST [2] 
* URL: https://huggingface.co [2] 
* Send Headers: Toggle this On.
* Add a header parameter where Name is Content-Type and Value is application/json. [2, 6] 
* Send Body: Toggle this On. [6] 
* Body Content Type: Select JSON. [6] 
* Specify Body: Select Using JSON/Raw Parameters. [2] 
* JSON/Raw Parameters: Paste the standard FLUX input format:

{
  "inputs": "An astronaut riding a horse on Mars, cinematic lighting, highly detailed"
}

(You can replace the prompt text with an expression from a previous node, like {{ $json.myPrompt }}). [6] 
* 

------------------------------
## 3. Crucial Step: Response Settings (Binary Mode)
Because FLUX returns raw image bytes rather than text or JSON data, you must instruct n8n to output the result as an image file: [1] 

   1. Scroll down to the bottom of the HTTP Request node options and open Response (or Options / Formatting depending on your n8n version).
   2. Set Response Format to File or Binary.
   3. For Put Output File in Field (or Binary Property), type a custom name like data or image. [1] 

When you execute the node, it will successfully contact Hugging Face using your bearer token, pass the prompt, and output a downloadable visual image block directly into the n8n UI environment. [2, 6] 
Would you like help passing a dynamic prompt from a previous trigger node (like an AirTable, Webhook, or Form node), or do you need assistance routing the generated image output to a destination like Google Drive or Discord?

[1] [https://discuss.huggingface.co](https://discuss.huggingface.co/t/newbie-needs-help-with-returned-data-type-is-not-an-image/173765)
[2] [https://www.youtube.com](https://www.youtube.com/watch?v=qeYgROvh1gY&t=129)
[3] [https://docs.n8n.io](https://docs.n8n.io/integrations/builtin/credentials/httprequest)
[4] [https://community.n8n.io](https://community.n8n.io/t/using-a-bearer-token-with-http-request-api-calls/25264)
[5] [https://n8nautomation.cloud](https://n8nautomation.cloud/blog/n8n-hugging-face-integration-workflows)
[6] [https://www.youtube.com](https://www.youtube.com/watch?v=nqQOYjLAuaM&t=159)
