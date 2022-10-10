
# Session 04: Deployment Demo 

A simple deployment demo project just to understand working of gradio and streamlit.

### Tools for deployment protoype
  - Gradio
  - Streamlit



# Docker
There are two scripts for docker building. One for training it includes complete dependencies for the training pipeline. And other for the deployment demo it only includes packages related to inference. 

**Note:: In deployment docker model is packed inside docker image. After training copy the scripted model to root dir so while building deployment docker it will add the file. Change the path of model in `demo_scripted.yaml`**



## Train
  - BUILD -> `docker/build.train`
  - TRAIN -> `python3 src/train.py experiment=cifar`
  

## Deployment
  - BUILD -> `docker/build`
  - DEMO -> `docker run -t -p 8080:8080  aditdoshi3333/gradio_deployment_demo:latest`
 


  




