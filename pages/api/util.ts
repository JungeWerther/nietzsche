import { OpenAI } from "langchain/llms";
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { PromptTemplate } from "langchain/prompts";
import { OpenAIChat } from 'langchain/llms';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT = PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are the reincarnation of Friedrich Nietzsche. You are a philosopher and a writer. You have read his collected work and are able to answer all questions regarding it.
  When a user asks you a question, you are allowed to respond cynically and with contempt.
  If relevant, you will respond with one and only one quote from your (Nietzsche's) collected work, which is provided in the context.
  Never make up quotes if you can't find anything.
  Display all quotes in bold and highlight them on a new line.
  Don't ask a similar question twice. Instead, come up with a new question.
  In addition, you love hugging horses and are a vegan. Only mention horses and veganism when the user explicitly asks you about it.
  Always answer the question. If you don't have an answer, ask a question of your own.
  Above all, keep the conversation coherent. Begin: paraphrase the user's question, relevant quote from your work, answer.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);


export const makeChain = (vectorstore: HNSWLib, onTokenStream?: (token: string) => void) => {
  
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });

  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.5,
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
      ? CallbackManager.fromHandlers({
          async handleLLMNewToken(token) {
            onTokenStream(token);
            console.log(token);
          },
        })
      : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
  });
}




// export const makeChain = (
//   vectorstore: PineconeStore,
//   onTokenStream?: (token: string) => void,
// ) => {
//   const questionGenerator = new LLMChain({
//     llm: new OpenAIChat({ temperature: 0.2 }),
//     prompt: CONDENSE_PROMPT,
//   });
  
  // const docChain = loadQAChain(
  //   new OpenAIChat({
  //     temperature: 0.5,
  //     modelName: 'gpt-4', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
  //     streaming: Boolean(onTokenStream),
  //     callbackManager: onTokenStream
  //       ? CallbackManager.fromHandlers({
  //           async handleLLMNewToken(token) {
  //             onTokenStream(token);
  //             console.log(token);
  //           },
  //         })
  //       : undefined,
  //   }),
  //   { prompt: QA_PROMPT },
  // );

//   return new ChatVectorDBQAChain({
//     vectorstore,
//     combineDocumentsChain: docChain,
//     questionGeneratorChain: questionGenerator,
//     returnSourceDocuments: true,
//     k: 3, //number of source documents to return
//   });
// };
