from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    """Main page with options for different functionalities"""
    return render_template('index.html')

@app.route('/upload')
def upload_and_analyze():
    """Handle PDF upload and analysis page"""
    return render_template('upload.html')

@app.route('/about')
def about_page():
    """Display the about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)