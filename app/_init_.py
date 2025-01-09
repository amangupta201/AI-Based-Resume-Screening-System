from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    # Your app setup here
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Create 'uploads' folder if not present

    # Register Blueprints
    from app.routes import main

    app.register_blueprint(main)

    return app

# Directly run the app
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
