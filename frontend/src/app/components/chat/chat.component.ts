import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { finalize } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { ChatMessage } from '../../models/api.models';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatButtonModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressSpinnerModule,
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent {
  @Input() videoId = '';
  @Input() collectionName = '';
  @Input() disabled = false;

  readonly form = new FormGroup({
    question: new FormControl('', [Validators.required]),
  });

  loading = false;
  error = '';
  messages: ChatMessage[] = [];

  constructor(private readonly apiService: ApiService) {}

  get sendDisabled(): boolean {
    return this.disabled || this.loading || this.form.invalid || !this.videoId;
  }

  ask(): void {
    if (this.sendDisabled) {
      return;
    }

    const question = this.form.controls.question.value ?? '';
    this.messages = [...this.messages, { sender: 'user', text: question }];
    this.form.reset();
    this.error = '';
    this.loading = true;

    this.apiService
      .askQuestion({
        question,
        video_id: this.videoId || undefined,
        collection_name: this.collectionName || undefined,
      })
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (res) => {
          this.messages = [
            ...this.messages,
            {
              sender: 'assistant',
              text: res.answer,
              sources: res.sources,
            },
          ];
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Chat request failed';
          const status = err?.status ? `HTTP ${err.status}` : 'HTTP ?';
          this.error = `${status}: ${detail}`;
        },
      });
  }
}
