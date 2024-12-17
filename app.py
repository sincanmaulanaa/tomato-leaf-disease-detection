from flask import Flask, render_template, request, redirect, url_for, session, flash
from connection import client
from detection import predict
import os
import bcrypt
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'tomatos'  # Replace with your secret key

# Connect to MongoDB
db = client['tomato']  # Replace with your database name
users_collection = db['users']  # Replace with your collection name
detections_collection = db['detections']  # New collection for storing detections

@app.route('/')
def home():
    return render_template('index.html')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the username or email already exists
        if users_collection.find_one({'username': username}):
            flash('Username already used')
            return redirect(url_for('register'))
        if users_collection.find_one({'email': email}):
            flash('Email already used')
            return redirect(url_for('register'))
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert user data into MongoDB
        users_collection.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        })
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Find the user in the database
        user = users_collection.find_one({'username': username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Password matches
            session['username'] = username
            return redirect(url_for('home'))
        else:
            # Invalid credentials
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        disease_name, probability = predict(file_path)
        
        # Store detection result in the database
        detections_collection.insert_one({
            'username': session['username'],
            'file_path': file_path,
            'disease_name': disease_name,
            'probability': probability,
            'date': datetime.now()
        })
        
        return render_template('result.html', disease_name=disease_name, probability=probability)

@app.route('/history')
@login_required
def history():
    user_detections = detections_collection.find({'username': session['username']})
    return render_template('history.html', detections=user_detections)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)