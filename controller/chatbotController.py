from flask import Blueprint, render_template, request, jsonify
from controller import CoreController

class ChatbotController:
    def __init__(self):
        self.view_base = 'chatbot'
        self.core_controller = CoreController()  # Initialize CoreController once

    def index(self):
        # Renders the main chatbot HTML page
        return render_template(f'{self.view_base}.html')
    
    def send_message(self):
        # Log request metadata for debugging
        print("Request Headers:", dict(request.headers))
        print("Raw Request Data:", request.data.decode("utf-8"))

        # Step 1: Parse JSON input
        try:
            data = request.get_json(force=True)
            print("Parsed JSON:", data)
        except Exception as e:
            print("Failed to parse JSON:", str(e))
            return jsonify({'error': 'Invalid JSON format'}), 400

        # Step 2: Extract and validate message
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Step 3: Generate model response
        try:
            response = self.core_controller.conv(message)
            print("Model Response:", response)
        except Exception as e:
            print(f"Model error: {str(e)}")
            return jsonify({'error': 'Error generating response from model'}), 500

        # Step 4: Return the response
        return jsonify({'response': response})
