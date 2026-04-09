pipeline {
    agent { label 'built-in' }

    environment {
        ENV_SOURCE = '/opt/config/object-detection/.env'
    }

    stages {

        stage('Prepare Workspace') {
            steps {
                cleanWs()
            }
        }

        // ❗ TIDAK PERLU checkout manual
        // Jenkins otomatis clone dari SCM

        stage('Prepare Environment File') {
            steps {
                sh '''
                if [ -f "$ENV_SOURCE" ]; then
                    cp $ENV_SOURCE .env
                else
                    echo ".env source not found!"
                    exit 1
                fi
                '''
            }
        }

        stage('Ensure Network') {
            steps {
                sh '''
                docker network inspect shared_network >/dev/null 2>&1 || \
                docker network create shared_network
                '''
            }
        }

        stage('Stop Old Containers') {
            steps {
                sh '''
                docker compose down --remove-orphans || true
                docker rm -f object-detection || true
                '''
            }
        }

        stage('Build Containers') {
            steps {
                sh '''
                docker compose build --no-cache
                '''
            }
        }

        stage('Run Containers') {
            steps {
                sh '''
                docker compose up -d
                '''
            }
        }

        stage('Show Status') {
            steps {
                sh '''
                echo "=== Container Status ==="
                docker ps | grep object-detection || echo "Container not running"
                '''
            }
        }
    }

    post {
        success {
            echo 'Deploy sukses 🚀'
        }
        failure {
            echo 'Deploy gagal ❌'
        }
        always {
            echo 'Pipeline completed.'
        }
    }
}