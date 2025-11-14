# GIMP LGI Plugin Makefile

CC = gcc
CFLAGS = $(shell gimptool-2.0 --cflags) -I/usr/local/include
LDFLAGS = $(shell gimptool-2.0 --libs) -L/usr/local/lib -llgi_ffi

PLUGIN = file-lgi

all: $(PLUGIN)

$(PLUGIN): file-lgi.c
	$(CC) $(CFLAGS) -o $(PLUGIN) file-lgi.c $(LDFLAGS)

install: $(PLUGIN)
	gimptool-2.0 --install-bin $(PLUGIN)

uninstall:
	gimptool-2.0 --uninstall-bin $(PLUGIN)

clean:
	rm -f $(PLUGIN)

test: install
	@echo "Plugin installed. Test in GIMP:"
	@echo "  File -> Open -> select .lgi file"
	@echo "  File -> Export As -> save as .lgi"

.PHONY: all install uninstall clean test
