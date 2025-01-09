# Launch of the Phi-3.5-vision container

The Phi-3.5-vision container is the API used to chat with the Phi-3.5-vision VLM.
This container is an implementation of Phi-3.5-vision-instruct-onnx, using the [ONNX Runtime generate() API](https://onnxruntime.ai/docs/genai/).
This implementation is optimised so that the inference time is reduced.

The API is a compatible OpenAI API. It is automatically generated with [SimpleAI](https://github.com/lhenault/simpleAI).

This implementation is composed of 2 parts:

* The VLM container (containing the VLM, on port 50051)
* The gRCP server (server on port 10999, retrieving requests and transmitting them to the VLM container)

This implementation allows you to launch the gRCP server anywhere.


### Build the docker image and launch the container

`docker compose up --build --force-recreate -d`

### Port

The production port is `10999`

### Available endpoints

To see the different endpoints available, check [http://localhost:10999/v1/docs#/](http://localhost:10999/v1/docs#/)

(/v1/completions, /v1/edits, /v1/embeddings are not supported yet)