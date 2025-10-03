import { Injectable } from '@nestjs/common';
import { BedrockEmbeddings, ChatBedrockConverse } from "@langchain/aws";
import { IterableReadableStream } from "@langchain/core/dist/utils/stream";
import { AIMessageChunk } from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatDto } from "./chat.dto";
import { TYPES } from "./message.dto";
import { RAGNeededPrompt, CourseDataDecisionPrompt } from "./prompts/templates";

import {
  BedrockRuntimeClient,
  ConverseCommand,
  type ConverseCommandInput,
} from "@aws-sdk/client-bedrock-runtime";
import type { FromSchema } from "json-schema-to-ts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
const fs = require("node:fs/promises");
const path = require("path");


// Helper funcs
function extractCourseIdsFromText(text: string): string[] {
  if (!text) return [];
  // Example matches: CS101, CS-101, CS 101, MATH205A, EE-200B, ΗΥ-120, ΗΥ120
  // \p{L} = any letter (Unicode), use /u flag
  const re = /\b([\p{L}]{2,}\s?-?\s?\d{2,3}[\p{L}]?)\b/gu;
  const hits = new Set<string>();
  for (const m of text.normalize('NFKC').matchAll(re)) {
    // normalize: uppercase (locale-aware), collapse spaces, normalize hyphens
    const norm = m[1]
      .normalize('NFKC')
      .toLocaleUpperCase('el-GR')   // handles Greek correctly; fine for Latin too
      .replace(/\s+/g, '')
      .replace(/-+/g, '-');
    hits.add(norm);
  }
  return [...hits];
}

function collectSeenCourseIds(
  convo: Array<{ role: "system" | "user" | "assistant"; content: string }>
): string[] {
  const set = new Set<string>();
  for (const m of convo) {
    for (const id of extractCourseIdsFromText(m.content)) set.add(id);
  }
  return [...set];
}

@Injectable()
export class GenAIService {
  static GENAI_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0";
  static EMBEDDINGS_MODEL = "amazon.titan-embed-text-v2:0";
  static VECTORSTORE_DIR = "data/vectorstores"

  getGenAIModel() {
    return new ChatBedrockConverse({
      region: process.env.AWS_REGION ?? "us-east-1",
      model: GenAIService.GENAI_MODEL,
      temperature: 0.0,
      credentials: {
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY ?? "",
        accessKeyId: process.env.AWS_ACCESS_KEY_ID ?? "",
        sessionToken: process.env.AWS_SESSION_TOKEN ?? "",
      },
    });
  }

  /**
   * Gets the embeddings model instance.
   * @returns The BedrockEmbeddings model instance.
   */
  getEmbeddingsModel() {
    return new BedrockEmbeddings({
      maxRetries: 0,
      region: process.env.AWS_REGION ?? "us-east-1",
      model: GenAIService.EMBEDDINGS_MODEL,
      credentials: {
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY ?? "",
        accessKeyId: process.env.AWS_ACCESS_KEY_ID ?? "",
        sessionToken: process.env.AWS_SESSION_TOKEN ?? "",
      },
    });
  }

  /**
   * Prompts the Generative AI model with a message.
   * @param message The message to send to the model.
   * @returns The response text from the model or null if an error occurs.
   */
  async prompt(message: string): Promise<string | null> {
    try {
      const model = this.getGenAIModel();
      const response = await model.invoke(message);

      return response.text;
    } catch ( err ) {
      console.log("Error in GenAIService.prompt:", err);
      return null;
    }
  }

  /**
   * Streams a response from the Generative AI model based on a message.
   * @param message The message to send to the model.
   * @returns An iterable stream of AI message chunks or null if an error occurs.
   */
  async stream(message: string): Promise<IterableReadableStream<AIMessageChunk> | null> {
    try {
      const model = this.getGenAIModel();
      return await model.stream(message);
    } catch ( err ) {
      console.log("Error in GenAIService.stream:", err);
      return null;
    }
  }

  /**
   * Chats with the Generative AI model using a ChatDto object.
   * @param chatDto The ChatDto containing messages to send to the model.
   * @returns The response text from the model or null if an error occurs.
   */
  async chat(chatDto: ChatDto): Promise<string | null> {
    try {
      const model = this.getGenAIModel();
      const input: any[] = [];

      for (const message of chatDto.messages) {
        input.push({role: message.type == TYPES.HUMAN ? 'user' : 'assistant', content: message.text});
      }

      const response = await model.invoke(input);

      return response.text;
    } catch ( err ) {
      console.log("Error in GenAIService.chat:", err);
      return null;
    }
  }

  /**
   * Generates embeddings for a given message.
   * @param message The message to generate embeddings for.
   * @returns A string representation of the first embedding or null if an error occurs.
   */
  async embeddings(message: string): Promise<string | null> {
    try {
      const model = this.getEmbeddingsModel();
      const res = await model.embedQuery(message);

      // Output only the first embedding for brevity
      return '[ ' + res[0].toString() + ', ... ]';
    } catch ( err ) {
      console.log("Error in GenAIService.embeddings:", err);
      return null;
    }
  }

  /**
   * Indexes documents from a specified directory into a vector store.
   * @param name The name of the vector store to save.
   * @param srcDir The source directory containing documents to index.
   * @returns The HNSWLib vector store or null if an error occurs.
   */
  async index(name: string, srcDir: string): Promise<HNSWLib | null> {
    try {
      const embeddingsModel = this.getEmbeddingsModel();
      const srcDirPath = process.cwd() + '/data/' + srcDir;
      const files = await fs.readdir(srcDirPath);
      const docs: Document[] = [];

      for (const file of files) {
        const docLoader = new TextLoader(process.cwd() + '/data/' + srcDir + "/" + file);
        const loadedDoc = await docLoader.load();
        docs.push(...loadedDoc);
      }

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, chunkOverlap: 200
      });
      const allSplits = await splitter.splitDocuments(docs);

      const vectorStore = await HNSWLib.fromDocuments(allSplits, embeddingsModel);
      await vectorStore.save(process.cwd() + "/" + GenAIService.VECTORSTORE_DIR + "/" + name);
      return vectorStore;

    } catch (err) {
      console.log("Error in GenAIService.rag:", err);
      return null;
    }
  }

    /**
   * Indexes documents from a specified directory into a vector store.
   * @param name The name of the vector store to save.
   * @param srcDir The source directory containing documents to index.
   * @returns The HNSWLib vector store or null if an error occurs.
   */
    async indexJSON(name: string, srcDir: string): Promise<HNSWLib | null> {
      try {
        const INDEX_FIELDS = ["general_info", "course_objectives", "course_content", "student_evaluation"]
        const embeddingsModel = this.getEmbeddingsModel();
        const srcDirPath = process.cwd() + '/data/' + srcDir;
        let files = await fs.readdir(srcDirPath);
        files = files.filter((f: string) => f.toLowerCase().endsWith(".json"));
        const docs: Document[] = [];
  

        for (const file of files) {
          const filePath = process.cwd() + '/data/' + srcDir + "/" + file;
          const jsonData = await fs.readFile(filePath, "utf-8");
          const loadedDoc = JSON.parse(jsonData);
          const chunks = INDEX_FIELDS.map(field => {
            if (loadedDoc[field]) {
              return new Document({
                pageContent: typeof loadedDoc[field] === 'object' 
                  ? JSON.stringify(loadedDoc[field]) 
                  : loadedDoc[field],
                metadata: {
                  courseId: loadedDoc["general_info"]["course_id"]
                }
              });
            }
            return null;
          });

          docs.push(...chunks.filter(c => c !== null) as Document[]);
        }
  
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000, chunkOverlap: 200
        });
        const allSplits = await splitter.splitDocuments(docs);

        // Cleanup
        allSplits.forEach(doc => {
          delete doc.metadata.loc;
        });
  
        const vectorStore = await HNSWLib.fromDocuments(allSplits, embeddingsModel);
        await vectorStore.save(process.cwd() + "/" + GenAIService.VECTORSTORE_DIR + "/" + name);
        return vectorStore;
  
      } catch (err) {
        console.log("Error in GenAIService.rag:", err);
        return null;
      }
    }

  async getCourseData(id: string): Promise<string> {
    try {
      const path = process.cwd() + '/data/combined_pdfs/combined.json';
      const courses = JSON.parse(await fs.readFile(path, "utf-8"));
      const course = courses.find((c: any) => c.general_info.course_id === id || c.general_info.course_id.slice(-3) === id.slice(-3));
      if (!course) {
        throw new Error(`Course with id ${id} not found`);
      }
      return JSON.stringify(course, null, 2);
    } catch (err) {
      console.log("Error in GenAIService.getCourseData:", err);
      return err.toString();
    }
  } 

  async ragIndexed(
    conversation: Array<{ role: "system" | "user" | "assistant"; content: string }>,
    collectionName: string
  ): Promise<string | null> {
    try {
      const llm = this.getGenAIModel();
      const embeddingsModel = this.getEmbeddingsModel();
  
      // 1) Decision prompt: does this need RAG?
      const decisionPrompt = [
        { role: "system", content: RAGNeededPrompt },
        {
          role: "user",
          content:
            "Conversation (most recent last):\n" +
            conversation.map(m => `[${m.role}] ${m.content}`).join("\n")
        }
      ];
      const lastUserIdx = [...conversation].map(m => m.role).lastIndexOf("user");
      const history = conversation.slice(0, lastUserIdx); // prior turns
      const lastUserMsg = conversation[lastUserIdx]?.content ?? "";
  
      const decisionRaw = await llm.invoke(decisionPrompt as any);
  
      let needRag = true;
      let rewrittenQuestion = "";
  
      try {
        const parsed = JSON.parse(
          typeof decisionRaw.content === "string" ? decisionRaw.content : String(decisionRaw.content)
        );
        needRag = !!parsed.need_rag;
        rewrittenQuestion = String(parsed.rewritten_question || "").trim();
      } catch {
        const lastUser = [...conversation].reverse().find(m => m.role === "user");
        needRag = true;
        rewrittenQuestion = lastUser?.content ?? "";
      }
  
      console.log("Rewritten question: " + rewrittenQuestion);
      console.log("RAG needed: " + needRag);
  
      if (!needRag) {
        const directAnswer = await llm.invoke(conversation as any);
        return typeof directAnswer.content === "string"
          ? directAnswer.content
          : String(directAnswer.content);
      }
  
      // 2) RAG path: load vector store
      const vectorStore = await HNSWLib.load(
        process.cwd() + "/" + GenAIService.VECTORSTORE_DIR + "/" + collectionName,
        embeddingsModel
      );
  
      // 3) Retrieve docs
      const retrievedDocs = await vectorStore.similaritySearch(rewrittenQuestion || lastUserMsg, 10);
  
      // 4) Gather retrieved + seen course ids
      const retrievedCourseIds = Array.from(
        new Set(
          retrievedDocs
            .map(d => (d.metadata?.courseId ?? d.metadata?.course_id ?? ""))
            .filter(Boolean)
            .map((s: string) => String(s).toUpperCase().replace(/\s+/g, '').replace(/-+/g, '-'))
        )
      );
  
      const seenCourseIds = collectSeenCourseIds(conversation);
      console.log("Seen course IDs in conversation:", seenCourseIds);
      console.log("Retrieved course IDs:", retrievedCourseIds);
  
      // 5) Ask Claude if we should call getCourseData (STRICT JSON)
      const courseDecisionPrompt = [
        { role: "system", content: CourseDataDecisionPrompt },
        {
          role: "user",
          content:
            [
              "Recent conversation (most recent last):",
              history.slice(-8).map(m => `[${m.role}] ${m.content}`).join("\n"),
              `[user] ${lastUserMsg}`,
              "",
              `seen_course_ids: ${JSON.stringify(seenCourseIds)}`,
              `retrieved_course_ids: ${JSON.stringify(retrievedCourseIds)}`,
              "",
              "Your job: Decide if getCourseData should be called now."
            ].join("\n")
        }
      ];
  
      let shouldCallCourseData = false;
      let chosenCourseId: string | null = null;
  
      try {
        const decision = await llm.invoke(courseDecisionPrompt as any);
        const parsed = JSON.parse(
          typeof decision.content === "string" ? decision.content : String(decision.content)
        );
        shouldCallCourseData = !!parsed.should_call;
        chosenCourseId = (parsed.course_id ?? null) ? String(parsed.course_id) : null;
        console.log("CourseData decision:", parsed);
      } catch (e) {
        console.log("CourseData decision parse failed, defaulting to not calling.", e);
        shouldCallCourseData = false;
        chosenCourseId = null;
      }

      console.log(`CourseData decision: shouldCall=${shouldCallCourseData}, chosenId=${chosenCourseId}`);
      

      // 6) Build RAG context (optionally with course data)
      const docsContent = retrievedDocs
        .map(doc => "Course:" + (doc.metadata?.courseId ?? doc.metadata?.course_id ?? "UNKNOWN") + "\n" + doc.pageContent)
        .join("\n");
  
      for (const doc of retrievedDocs) console.log(doc.metadata);
  
      const ragMessages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
        ...history.slice(-8),
        { role: "system", content: `Courses:\n${docsContent}` }
      ];
  
      if (shouldCallCourseData && chosenCourseId) {
        try {
          const courseJson = await this.getCourseData(chosenCourseId);
          ragMessages.push({
            role: "system",
            content: `CourseData for ${chosenCourseId}:\n${courseJson}`
          });
        } catch (e) {
          console.log("Error calling getCourseData; continuing without it.", e);
        }
      }
  
      ragMessages.push({ role: "user", content: lastUserMsg });
  
      const response = await llm.invoke(ragMessages as any);
      return typeof response.content === "string" ? response.content : String(response.content);
    } catch (err) {
      console.log("Error in GenAIService.rag:", err);
      return null;
    }
  }

  /**
   * Performs a retrieval-augmented generation (RAG) process.
   * @param message The query message to process.
   * @returns The generated answer or null if an error occurs.
   */
  async rag(message: string): Promise<string | null> {
    try {
      // Create the models (LLM, embeddings)
      const llm = this.getGenAIModel();
      const embeddings = this.getEmbeddingsModel();

      // Load the document and create the embeddings
      const vectorStore = new MemoryVectorStore(embeddings);

      const docLoader = new TextLoader(process.cwd() + '/data/AliceAdventuresInWonderland.txt');
      const docs = await docLoader.load();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, chunkOverlap: 200
      });
      const allSplits = await splitter.splitDocuments(docs);
      await vectorStore.addDocuments(allSplits);

      // Prepare the Graph to process the incoming query
      const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

      const InputStateAnnotation = Annotation.Root({
        question: Annotation<string>,
      });

      const StateAnnotation = Annotation.Root({
        question: Annotation<string>,
        context: Annotation<Document[]>,
        answer: Annotation<string>,
      });

      const retrieve = async (state: typeof InputStateAnnotation.State) => {
        const retrievedDocs = await vectorStore.similaritySearch(state.question);
        return { context: retrievedDocs };
      };

      const generate = async (state: typeof StateAnnotation.State) => {
        const docsContent = state.context.map(doc => doc.pageContent).join("\n");
        const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
        const response = await llm.invoke(messages);
        return { answer: response.content };
      };

      const graph = new StateGraph(StateAnnotation)
          .addNode("retrieve", retrieve)
          .addNode("generate", generate)
          .addEdge("__start__", "retrieve")
          .addEdge("retrieve", "generate")
          .addEdge("generate", "__end__")
          .compile();

      // Invoke the graph and get the result
      const result = await graph.invoke({ question: message });

      console.log(`\nCitations: ${result.context.length}`);
      console.log(result.context.slice(0, 2));

      return result.answer;
    } catch ( err ) {
      console.log("Error in GenAIService.rag:", err);
      return null;
    }
  }


  async pdfToStructuredJson(): Promise<string> {
    const client = new BedrockRuntimeClient({
      region: process.env.AWS_REGION || "us-east-1",
    });

    const schemaPath = path.join(process.cwd(), "src", "schemas", "course_schema.json");
    const schema = JSON.parse(await fs.readFile(schemaPath, "utf-8"));
    const toolName = "output";


    let files = await fs.readdir(path.join(process.cwd(), "data", "pdfs"));
    files.filter((f: string) => f.toLowerCase().endsWith(".pdf"));

    for (const file of files) {
      const pdfPath = path.join(process.cwd(), "data", "pdfs", file);
      console.log(pdfPath);
      const pdfBytes = await fs.readFile(pdfPath);

      const input: ConverseCommandInput = {
        modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages: [
          {
            role: "user",
            content: [
              {
                text:
                  `Extract the structured data from this PDF. 
  Return the result by "calling" the tool named "${toolName}" using parameters that EXACTLY match the input schema. 
  Do not include prose outside the tool call.`,
              },
              {
                document: {
                  format: "pdf",
                  name: "pdf1",
                  source: { bytes: pdfBytes },
                },
              },
              { text: `Schema (for reference):\n${JSON.stringify(schema, null, 2)}` },
            ],
          },
        ],

        toolConfig: {
          tools: [
            {
              toolSpec: {
                name: toolName,
                description: "Return the extracted data in this schema.",
                inputSchema: {
                  json: schema,
                },
              },
            },
          ],
        },

        inferenceConfig: { maxTokens: 2000, temperature: 0.05 },
      };

      const resp = await client.send(new ConverseCommand(input));

      const contentBlocks = resp.output?.message?.content ?? [];

      // Claude returns something like: { toolUse: { name, input: {...} } }
      const toolUseBlock = contentBlocks.find((b: any) => b?.toolUse)?.toolUse;
      if (!toolUseBlock) {
        // Fallback: if the model replied as plain text, try to parse JSON text
        const text = contentBlocks.find((b: any) => b?.text)?.text?.trim() ?? "";
        if (!text) throw new Error("No toolUse block and no text response from model.");
        // best-effort JSON extraction
        const start = text.indexOf("{");
        const end = text.lastIndexOf("}");
        const jsonStr = start >= 0 && end >= 0 ? text.slice(start, end + 1) : text;
        return jsonStr;
      }

      // toolUseBlock.input already matches your schema
      await fs.writeFile(pdfPath + ".json", JSON.stringify(toolUseBlock.input) + "\n", function (err) {
        if (err) throw err;
        console.log('Saved!');
      });
    }
    return "OK";
  }
}
