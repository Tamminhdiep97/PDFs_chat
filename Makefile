init:
	@echo "init step for CHATBOT SYSTEM"


build-dev:
	@echo "Build docker dev image"
	docker compose -f docker_setup/docker-compose.dev.yml build --no-cache


up-dev:
	@echo "CHATBOT dev up"
	docker compose -f docker_setup/docker-compose.dev.yml up -d


log-dev:
	@echo "CHATBOT dev log"
	docker compose -f docker_setup/docker-compose.dev.yml logs -f --tail 500


stop-dev:
	@echo "CHATBOT dev stop"
	docker compose -f docker_setup/docker-compose.dev.yml stop
