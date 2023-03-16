from forecasting import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host="192.168.0.103", port="7777")



# A very simple Flask Hello World app for you to get started with...

# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello from Flask!'

#flask_app,py