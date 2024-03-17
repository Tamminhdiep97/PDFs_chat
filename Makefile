init:
	@echo "init step for CHATBOT SYSTEM"


build-dev:
	@echo "Build docker dev image"
	docker-compose -f docker_setup/docker-compose.dev.yml build


up-dev:
	@echo "CHATBOT dev up"
	docker-compose -f docker_setup/docker-compose.dev.yml up
