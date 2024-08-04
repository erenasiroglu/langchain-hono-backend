import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import path from "path";
import { promises as fs } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

const app = new Hono()

const getTextFile = async () => {

  const filePath = path.join(__dirname, "../data/langchain-test.txt");

  const data = await fs.readFile(filePath, "utf-8");

  return data;
}

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

// Vector Db

let vectorStore : MemoryVectorStore;

app.get('/loadTextEmbeddings', async (c) => {

  const text = await getTextFile();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      separators:['\n\n', '\n', ' ', '', '###'],
      chunkOverlap: 50
    });

    const output = await splitter.createDocuments([text])

  const embeddings = new OllamaEmbeddings({
    model: "gemma2:2b", // gemma2:2b
    baseUrl: "http://localhost:11434", // default value
    requestOptions: {
      useMMap: true, // use_mmap 1
      numThread: 6, // num_thread 6
      numGpu: 1, // num_gpu 1
    },
  });

    vectorStore = await MemoryVectorStore.fromDocuments(output, embeddings);

    const response = {message: "Text embeddings loaded successfully."};

    return c.json(response);
})

const port = 3002
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
