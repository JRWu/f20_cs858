all:
	docker-compose up --build -d

shell: force
	docker-compose exec f20_cs858 bash

force: