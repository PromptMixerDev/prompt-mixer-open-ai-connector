import OpenAI from 'openai';
import { config } from './config.js';
import { ChatCompletion } from 'openai/resources';
import * as fs from 'fs';

const API_KEY = 'API_KEY';
const TOOL_CALLS_FINISH_REASON = 'tool_calls';

interface Message {
  role: string;
  content: Array<{ type: string; text?: string; image_url?: { url: string } }>;
}

interface Completion {
  Content: string | null;
  Error?: string | undefined;
  TokenUsage: number | undefined;
  FinishReason?: string;
}

interface ConnectorResponse {
  Completions: Completion[];
  ModelType: string;
}

interface ErrorCompletion {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  error: string;
  model: string;
  usage: undefined;
}

const mapToResponse = (
  outputs: Array<ChatCompletion | ErrorCompletion>,
  model: string,
): ConnectorResponse => {
  return {
    Completions: outputs.map((output) => {
      if ('error' in output) {
        return {
          Content: null,
          TokenUsage: undefined,
          Error: output.error,
        };
      } else if (output.choices[0]?.finish_reason === TOOL_CALLS_FINISH_REASON) {
        return {
          Content: JSON.stringify(output.choices[0]?.message?.tool_calls),
          TokenUsage: output.usage?.total_tokens,
          FinishReason: TOOL_CALLS_FINISH_REASON,
        }
      } else {
        return {
          Content: output.choices[0]?.message?.content,
          TokenUsage: output.usage?.total_tokens,
        };
      }
    }),
    ModelType: outputs[0].model || model,
  };
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mapErrorToCompletion = (error: any, model: string): ErrorCompletion => {
  const errorMessage = error.message || JSON.stringify(error);
  return {
    choices: [],
    error: errorMessage,
    model,
    usage: undefined,
  };
};

function isO1Model(model: string): boolean {
  return model.toLowerCase().startsWith('o1');
}

function encodeImage(imagePath: string): string {
  const imageBuffer = fs.readFileSync(imagePath);
  return Buffer.from(imageBuffer).toString('base64');
}

async function main(
  model: string,
  prompts: string[],
  properties: Record<string, unknown>,
  settings: Record<string, unknown>,
) {
  const openai = new OpenAI({
    apiKey: settings?.[API_KEY] as string,
  });

  const total = prompts.length;
  const { prompt, ...restProperties } = properties;
  const systemPrompt = (prompt ||
    config.properties.find((prop) => prop.id === 'prompt')?.value) as string;
  const messageHistory: Message[] = [];
  
  if (!isO1Model(model)) {
    messageHistory.push({ role: 'developer', content: [{ type: 'text', text: systemPrompt }] });
  }
  const outputs: Array<ChatCompletion | ErrorCompletion> = [];

  try {
    for (let index = 0; index < total; index++) {
      try {
        const userPrompt = prompts[index];
        const imageUrls = extractImageUrls(userPrompt);
        const messageContent: Message['content'] = [{ type: 'text', text: userPrompt }];

        for (const imageUrl of imageUrls) {
          if (imageUrl.startsWith('http')) {
            messageContent.push({
              type: 'image_url',
              image_url: {
                url: imageUrl,
              },
            });
          } else {
            const base64Image = encodeImage(imageUrl);
            messageContent.push({
              type: 'image_url',
              image_url: {
                url: `data:image/jpeg;base64,${base64Image}`,
              },
            });
          }
        }

        messageHistory.push({ role: 'user', content: messageContent });
        console.log(messageHistory);

        const chatCompletion = await openai.chat.completions.create({
          messages: messageHistory as unknown as [],
          model,
          ...restProperties,
        });

        outputs.push(chatCompletion);
        const assistantResponse =
          chatCompletion.choices[0].message.content || 'No response.';
        messageHistory.push({ role: 'assistant', content: [{ type: 'text', text: assistantResponse }] });
        console.log(
          `Response to prompt ${index + 1} of ${total}:`,
          chatCompletion,
        );
      } catch (error) {
        const completionWithError = mapErrorToCompletion(error, model);
        outputs.push(completionWithError);
      }
    }

    return mapToResponse(outputs, model);
  } catch (error) {
    console.error('Error in main function:', error);
    return { Error: error, ModelType: model };
  }
}


function extractImageUrls(prompt: string): string[] {
  const imageExtensions = ['.png', '.jpeg', '.jpg', '.webp', '.gif'];
  // Updated regex to match both http and local file paths
  const urlRegex = /(https?:\/\/[^\s]+|[a-zA-Z]:\\[^:<>"|?\n]*|\/[^:<>"|?\n]*)/g;
  const urls = prompt.match(urlRegex) || [];
  
  return urls.filter((url) => {
      const extensionIndex = url.lastIndexOf('.');
      if (extensionIndex === -1) {
          // If no extension found, return false.
          return false;
      }
      const extension = url.slice(extensionIndex);
      return imageExtensions.includes(extension.toLowerCase());
  });
}

export { main, config };
