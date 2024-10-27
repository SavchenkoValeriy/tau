import { join, dirname } from "node:path";

const STATIC = join(dirname(Bun.main), 'build');

const server = Bun.serve({
  port: 3000,
  async fetch(req: Request) {
    const u = new URL(req.url);
    const fileToServe = u.pathname == '/' ? 'index.html' : u.pathname;
    const f = Bun.file(join(STATIC, fileToServe));
    const exists = await f.exists();
    if (!exists)
      return new Response(`Not found: ${fileToServe}`, { status: 404 });

    const r = new Response(await f.arrayBuffer(), {
      headers: {
        "Content-Type": f.type,
      },
    });

    return r;
  },
});

console.log(`Listening on localhost:${server.port}`);
