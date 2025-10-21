import OpenAI from 'openai';
import { config } from './config.js';
import { ChatCompletion } from 'openai/resources';
import * as fs from 'fs';
import * as path from 'path';

const API_KEY = 'API_KEY';
const TOOL_CALLS_FINISH_REASON = 'tool_calls';

type MessageContent =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'file'; file: { filename?: string; file_data?: string; file_id?: string } };

interface Message {
  role: string;
  content: MessageContent[];
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

const SUPPORTED_FILE_TYPES: Record<
  string,
  { kind: 'image' | 'file'; mime: string; dataUrlPrefix: string }
> = {
  '.png': { kind: 'image', mime: 'image/png', dataUrlPrefix: 'data:image/png;base64,' },
  '.jpeg': { kind: 'image', mime: 'image/jpeg', dataUrlPrefix: 'data:image/jpeg;base64,' },
  '.jpg': { kind: 'image', mime: 'image/jpeg', dataUrlPrefix: 'data:image/jpeg;base64,' },
  '.webp': { kind: 'image', mime: 'image/webp', dataUrlPrefix: 'data:image/webp;base64,' },
  '.gif': { kind: 'image', mime: 'image/gif', dataUrlPrefix: 'data:image/gif;base64,' },
  '.pdf': { kind: 'file', mime: 'application/pdf', dataUrlPrefix: 'data:application/pdf;base64,' },
};

const LEADING_PUNCTUATION = /^[("'\[]+/;
const TRAILING_PUNCTUATION = /[)"'\],.?!:;]+$/;

const FILE_EXTENSION_PATTERN = '(?:png|jpeg|jpg|webp|gif|pdf)';
const FILE_REFERENCE_PATTERN = new RegExp(
  String.raw`https?:\/\/[^\s"'()<>]+|[a-zA-Z]:\\(?:[^\\\r\n"'|?<>]+\\)*[^\\\r\n"'|?<>]+\.${FILE_EXTENSION_PATTERN}|(?:\.\.?|~)?\/(?:[^\/\r\n"'|?<>]+\/)*[^\/\r\n"'|?<>]+\.${FILE_EXTENSION_PATTERN}|[^\s"'()<>]+\.${FILE_EXTENSION_PATTERN}`,
  'gi',
);

function encodeFileToDataUrl(filePath: string, prefix: string): string {
  const fileBuffer = fs.readFileSync(filePath);
  return `${prefix}${Buffer.from(fileBuffer).toString('base64')}`;
}

function isHttpUrl(value: string): boolean {
  return value.startsWith('http://') || value.startsWith('https://');
}

function sanitizeReference(reference: string): string {
  return reference.replace(LEADING_PUNCTUATION, '').replace(TRAILING_PUNCTUATION, '');
}

function getExtensionFromReference(reference: string): string {
  if (isHttpUrl(reference)) {
    try {
      const url = new URL(reference);
      return path.extname(url.pathname).toLowerCase();
    } catch {
      return path.extname(reference).toLowerCase();
    }
  }

  return path.extname(reference).toLowerCase();
}

function resolveLocalPath(reference: string): string {
  if (path.isAbsolute(reference)) {
    return reference;
  }

  if (/^[a-zA-Z]:\\/.test(reference)) {
    return path.normalize(reference);
  }

  return path.resolve(process.cwd(), reference);
}

interface FileReference {
  original: string;
  sanitized: string;
  extension: string;
  descriptor: { kind: 'image' | 'file'; mime: string; dataUrlPrefix: string };
  isRemote: boolean;
}

function extractFileReferences(prompt: string): FileReference[] {
  const candidates = prompt.match(FILE_REFERENCE_PATTERN) || [];
  const references = new Map<string, FileReference>();

  for (const candidate of candidates) {
    const sanitized = sanitizeReference(candidate);
    if (!sanitized) {
      continue;
    }

    const extension = getExtensionFromReference(sanitized);
    const descriptor = SUPPORTED_FILE_TYPES[extension];
    if (!descriptor) {
      continue;
    }

    const key = sanitized.toLowerCase();
    if (references.has(key)) {
      continue;
    }

    references.set(key, {
      original: candidate,
      sanitized,
      extension,
      descriptor,
      isRemote: isHttpUrl(sanitized),
    });
  }

  return Array.from(references.values());
}

async function buildAttachment(reference: FileReference): Promise<{
  attachment?: MessageContent;
  fallbackText?: string;
}> {
  if (reference.isRemote) {
    if (reference.descriptor.kind === 'image') {
      return {
        attachment: {
          type: 'image_url',
          image_url: {
            url: reference.sanitized,
          },
        },
      };
    }

    try {
      const response = await fetch(reference.sanitized);
      if (!response.ok) {
        return { fallbackText: reference.sanitized };
      }

      const buffer = Buffer.from(await response.arrayBuffer());
      const filename =
        (() => {
          try {
            const remotePathname = new URL(reference.sanitized).pathname;
            const baseName = path.basename(remotePathname);
            return baseName || `file${reference.extension}`;
          } catch {
            return `file${reference.extension}`;
          }
        })();

      return {
        attachment: {
          type: 'file',
          file: {
            filename,
            file_data: `${reference.descriptor.dataUrlPrefix}${buffer.toString('base64')}`,
          },
        },
      };
    } catch {
      return { fallbackText: reference.sanitized };
    }
  }

  try {
    const resolvedPath = resolveLocalPath(reference.sanitized);
    const stats = fs.statSync(resolvedPath);

    if (!stats.isFile()) {
      return { fallbackText: reference.sanitized };
    }

    const dataUrl = encodeFileToDataUrl(resolvedPath, reference.descriptor.dataUrlPrefix);

    if (reference.descriptor.kind === 'image') {
      return {
        attachment: {
          type: 'image_url',
          image_url: {
            url: dataUrl,
          },
        },
      };
    }

    return {
      attachment: {
        type: 'file',
        file: {
          filename: path.basename(resolvedPath),
          file_data: dataUrl,
        },
      },
    };
  } catch {
    return { fallbackText: reference.sanitized };
  }
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
        const fileReferences = extractFileReferences(userPrompt);
        const messageContent: Message['content'] = [{ type: 'text', text: userPrompt }];

        for (const reference of fileReferences) {
          const { attachment, fallbackText } = await buildAttachment(reference);

          if (attachment) {
            messageContent.push(attachment);
          }

          if (fallbackText) {
            messageContent.push({ type: 'text', text: fallbackText });
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


export { main, config };
