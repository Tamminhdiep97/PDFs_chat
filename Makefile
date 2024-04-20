init:
	@echo "init step for CHATBOT SYSTEM"
	cd docker_setup && cp .env.example .env


dev-init:
	make init
	cd docker_setup && cp docker-compose.dev.yml .docker-compose.yml


dev-build:
	@echo "Build docker dev image with no cache"
	make dev-init
	docker compose -f docker_setup/.docker-compose.yml build --no-cache


dev-up:
	@echo "CHATBOT dev env up"
	make dev-init
	docker compose -f docker_setup/.docker-compose.yml up -d


dev-log:
	@echo "CHATBOT dev log"
	make dev-init
	docker compose -f docker_setup/.docker-compose.yml logs -f --tail 500


dev-stop:
	@echo "CHATBOT dev stop"
	make  dev-init
	docker compose -f docker_setup/.docker-compose.yml stop


dev-restart:
	@echo "CHATBOT restart"
	make dev-stop
	make dev-up
