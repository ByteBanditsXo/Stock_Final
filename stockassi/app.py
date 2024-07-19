from flask import Flask
from config import Config
from models import db
from routes import configure_routes

app = Flask(__name__)
app.config.from_object(Config)

# Initialize the database with the app
db.init_app(app)

# Register routes
configure_routes(app)

# Initialize the database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])

