docker run -d --name splitfed-rabbit \
  -p 5672:5672 -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=admin \
  -e RABBITMQ_DEFAULT_PASS=admin \
  rabbitmq:3-management
curl -u admin:admin http://127.0.0.1:15672/api/overview
python server.py
    