interface Env {
  BEACH_DATA: R2Bucket;
}

export const onRequest: PagesFunction<Env> = async (ctx) => {
  const key = "data/" + (ctx.params.path as string[]).join("/");
  const rangeHeader = ctx.request.headers.get("range");

  // Pass Range header through to R2 so video seek/scrub works correctly.
  const options: R2GetOptions = rangeHeader ? { range: ctx.request.headers } : {};
  const obj = await ctx.env.BEACH_DATA.get(key, options);
  if (!obj) return new Response("Not found", { status: 404 });

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("etag", obj.httpEtag);

  if (rangeHeader && obj.range) {
    const r = obj.range as { offset: number; length: number };
    headers.set(
      "content-range",
      `bytes ${r.offset}-${r.offset + r.length - 1}/${obj.size}`
    );
    return new Response(obj.body, { status: 206, headers });
  }

  return new Response(obj.body, { headers });
};
