import { serve } from "@hono/node-server";
import { Hono } from "hono";
import path from "path";
import { promises as fs } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { Ollama } from "@langchain/community/llms/ollama";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { cors } from "hono/cors";

const app = new Hono();

app.use(
  "/*",
  cors({
    origin: "http://localhost:5173",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    exposeHeaders: ["Content-Length", "X-Kuma-Revision"],
    maxAge: 600,
    credentials: true,
  })
);

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "gemma2:2b",
});

const embeddings = new OllamaEmbeddings({
  model: "gemma2:2b",
  baseUrl: "http://localhost:11434",
  requestOptions: {
    useMMap: true,
    numThread: 6,
    numGpu: 1,
  },
});

const getTextFile = async () => {
  const filePath = path.join(__dirname, "../data/langchain-test.txt");
  const data = await fs.readFile(filePath, "utf-8");
  return data;
};

const loadPdfFile = async () => {
  const filePath = path.join(__dirname, "../data/trendspro-report.pdf");
  const loader = new PDFLoader(filePath);
  return await loader.load();
};

app.get("/", (c) => {
  return c.text("Hello Hono!");
});

let vectorStore: MemoryVectorStore;

app.get("/loadTextEmbeddings", async (c) => {
  const text = await getTextFile();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    separators: ["\n\n", "\n", " ", "", "###"],
    chunkOverlap: 50,
  });
  const output = await splitter.createDocuments([text]);
  vectorStore = await MemoryVectorStore.fromDocuments(output, embeddings);
  const response = { message: "Text embeddings loaded successfully." };
  return c.json(response);
});

app.get("/loadPdfEmbeddings", async (c) => {
  const documents = await loadPdfFile();
  vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
  const response = { message: "Text embeddings loaded successfully." };
  return c.json(response);
});

app.post("/ask", async (c) => {
  const { question } = await c.req.json();
  if (!vectorStore) {
    return c.json({ message: "Text embeddings not loaded yet." });
  }
  const prompt =
    PromptTemplate.fromTemplate(`You are a helpful AI assistant. Answer the following question based only on the provided context. If the answer cannot be derived from the context, say "I don't have enough information to answer that question." If I like your results I'll tip you $1000!

Context: {context}

Question: {question}

Answer: 
  `);
  const documentChain = await createStuffDocumentsChain({
    llm: ollama,
    prompt,
  });
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever: vectorStore.asRetriever({
      k: 3,
    }),
  });
  const response = await retrievalChain.invoke({
    question: question,
    input: "",
  });
  return c.json({ answer: response.answer });
});

const port = 3002;
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});
