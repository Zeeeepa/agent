export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AnthropicMessagesRequest {
  model: string;
  messages: AnthropicMessage[];
  max_tokens: number;
  metadata?: { user_id?: string };
  stop_sequences?: string[];
  stream?: boolean;
  system?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
}

export interface AnthropicContentBlock {
  type: 'text';
  text: string;
}

export interface AnthropicMessagesResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  content: AnthropicContentBlock[];
  model: string;
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | null;
  stop_sequence?: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface AnthropicStreamEventMessageStart {
  type: 'message_start';
  message: any;
}

export interface AnthropicStreamEventContentBlockDelta {
  type: 'content_block_delta';
  index: number;
  delta: { type: 'text_delta'; text: string };
}

export interface AnthropicStreamEventMessageDelta {
  type: 'message_delta';
  delta: { stop_reason: string | null };
  usage: { output_tokens: number };
}

export interface AnthropicStreamEventMessageStop {
  type: 'message_stop';
}

export interface AnthropicStreamEventError {
  type: 'error';
  error: { type: string; message: string };
}

export type AnthropicStreamEvent =
  | AnthropicStreamEventMessageStart
  | { type: 'content_block_start'; index: number; content_block: any }
  | AnthropicStreamEventContentBlockDelta
  | { type: 'content_block_stop'; index: number }
  | AnthropicStreamEventMessageDelta
  | AnthropicStreamEventMessageStop
  | { type: 'ping' }
  | AnthropicStreamEventError;

export interface AnthropicErrorResponse {
  type: 'error';
  error: { type: string; message: string };
}
