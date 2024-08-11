# Room Styler - Web API
FastAPI Web Project for Styling Living Rooms using InstructPix2Pix.

Each library used in the project is specified in the requirements.txt file. I provided a Docker file that uses a Python 3.9 image and uses this requirements file along with libmagic. To build the project, you can run the following command:

docker build -t room-styler .    


Normally, the Docker file is written to run the project on port 8000; however, you can specify another port (in the example, port 8080) by running the Docker container as follows:

docker run -p 8080:8000 room-styler 


If you run the container locally, you can find the service's endpoint here. I recommend you run the project on a machine with a GPU to make use of CUDA, as the inference time will be much shorter. Moreover, as I mentioned earlier, at start-up, the service will load the classifier and diffusion models; as the tensors are a few GBs, it may take a while. However, once the pipeline components are downloaded, the app will be up and running quickly. 

One final advice would be to give the prompts as “turn … into …” as the diffusion model was trained for such prompts.

