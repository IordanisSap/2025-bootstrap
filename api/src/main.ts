import { NestFactory } from '@nestjs/core';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';

function parseAllowlist(env?: string): string[] {
  return (env ?? '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
}

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // --- CORS ---
  const allowlist = parseAllowlist(process.env.FRONTEND_ORIGINS) || [
    'http://localhost:3001',
    // 'https://your-frontend.com',
  ];

  app.enableCors({
    origin: [
      'http://localhost:5173',
      'http://localhost:3000',
      'http://localhost:3001', // ← add this
      // 'https://your-frontend.com',
    ],
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true, // only if you need cookies/auth
    maxAge: 600,
  });
  // --------------

  const config = new DocumentBuilder()
    .setTitle('MentorMind API')
    .setDescription('The MentorMind API description')
    .setVersion('1.0')
    .build();
  const documentFactory = () => SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('schema', app, documentFactory);

  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();