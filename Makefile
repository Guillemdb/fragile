current_dir = $(shell pwd)

PROJECT = fragile
VERSION ?= latest

.POSIX:
.POSIX:
check:
	!(grep -R /tmp tests)
	flakehell lint ${PROJECT}
	pylint ${PROJECT}
	black --check ${PROJECT}

.PHONY: test
test:
	find -name "*.pyc" -delete
	pytest -s

.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm -it --network host -w /fragile --entrypoint python3 fragiletech/fragile:${VERSION} -m pytest


.PHONY: docker-build
docker-build:
	docker build --pull -t fragiletech/fragile:${VERSION} .

.PHONY: docker-push
docker-push:
	docker push fragiletech/fragile:${VERSION}