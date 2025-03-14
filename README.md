# AI playground

A simple project to experiment with llm models, prompts, langchain, RAG, chromaDB and in general on how to test this kind of applications

The project aims to be a chatbot replying technical questions for some OSS products (Gemfire, Greenplum, RabbitMQ).

The project is updating the knowledge of the model using RAG (with ChromaDB) collecting some info from the web and updating the model.

It is using some langchaing loader in order to load all the RabbitMQ website:
https://www.rabbitmq.com/

and the two pdfs for Gemfire and Greenplum official docs:

https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf

https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-greenplum/6/greenplum-database/greenplum-database.pdf

The project is using ollama in order to load a 'orca-mini' model running locally.

Probably in the future I will experiment with langgraph and huggingface too.



