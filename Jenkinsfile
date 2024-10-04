pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh '''#!/bin/bash
                echo 'Setting up Python virtual environment'

                # Create virtual environment if it doesn't exist
                if [ ! -d ".venv" ]; then
                    python3 -m venv .venv
                fi

                # Activate the virtual environment
                source .venv/bin/activate

                # Upgrade pip
                pip install --upgrade pip

                # Install dependencies
                pip install -r requirements.txt
                '''
            }
        }
        stage('Test') {
            steps {
                sh '''#!/bin/bash
                echo 'Running tests with pytest'

                # Activate the virtual environment
                source .venv/bin/activate

                # Run pytest
                pytest
                '''
            }
        }
        stage('Deploy') {
            steps {
                echo 'In this step, we deploy our project'
                echo 'Depending on the context, we may publish the project artifact or upload pickle files'
            }
        }
    }
}
