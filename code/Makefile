HOST_FILE := .current_host.tmp
PREV_HOST := $(shell cat $(HOST_FILE))
HOSTNAME := $(shell hostname)

all:
ifeq ($(HOSTNAME), $(PREV_HOST))
	@echo $(HOSTNAME)
	python2 setup.py build_ext -i
else
	@echo $(HOSTNAME)
	rm -rf build; rm -vf *.c *.cpp *.so *.html *.pyc
	python2 setup.py build_ext -i
	@echo $(HOSTNAME) > $(HOST_FILE)
endif

clean:
	rm -rf build; rm -vf *.c *.cpp *.so *.html *.pyc
