#include <stdio.h>
#include <string.h>
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

int main(void) {
    char buf[256];
    int error;
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);

    while(fgets(buf, sizeof(buf), stdin) != NULL) {
        error = luaL_loadstring(L, buf) || lua_pcall(L, 0, 0, 0);
        if(error) {
            fprintf(stderr, "%s\n", lua_tostring(L, -1));
            lua_pop(L, 1);
        }
    }
    lua_close(L);
    return 0;
}
