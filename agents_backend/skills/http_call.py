import httpx
timeout = 3600

async def http_call(uri, query):
    async with httpx.AsyncClient(timeout = timeout) as client:
        #print(query)
        resp = await client.post(uri, json = query)
        #print(resp)
        out = resp.json()["predictions"]
        """
        if not isinstance(out, list):
            return []
        else:
            if (len(out) > 0):
                data = get_cases_rag(out)
                return data
        """
        return out