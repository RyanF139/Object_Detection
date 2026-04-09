pipeline {
    agent { label 'built-in' }

    environment {
        ENV_SOURCE = '/opt/config/object-detection/.env'
        COMPOSE_DIR = 'docker'   // <-- ganti ini ke folder di repo yang berisi docker-compose.yml
    }

    stages {

        stage('Prepare Workspace') {
            steps {
                cleanWs()
            }
        }

        stage('Prepare Environment File') {
            steps {
                sh '''
                if [ -f "$ENV_SOURCE" ]; then
                    echo "Copying .env to workspace..."
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
                cd $COMPOSE_DIR
                if [ -f docker-compose.yml ]; then
                    docker compose down --remove-orphans || true
                    # Stop & remove old container by service name
                    docker ps -aq --filter "name=object-detection" | xargs -r docker rm -f || true
                else
                    echo "docker-compose.yml not found in $COMPOSE_DIR"
                fi
                '''
            }
        }

        stage('Build Containers') {
            steps {
                sh '''
                cd $COMPOSE_DIR
                if [ -f docker-compose.yml ]; then
                    docker compose build --no-cache
                else
                    echo "docker-compose.yml not found in $COMPOSE_DIR"
                    exit 1
                fi
                '''
            }
        }

        stage('Run Containers') {
            steps {
                sh '''
                cd $COMPOSE_DIR
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