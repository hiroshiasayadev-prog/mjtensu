import { MantineProvider, Stack, Title } from '@mantine/core';
import { BrowserRouter } from 'react-router-dom';

export function App() {
  return (
    <MantineProvider>
      <BrowserRouter>
        <main>
          <Stack p="md">
            <Title order={1}>mjtensu</Title>
          </Stack>
        </main>
      </BrowserRouter>
    </MantineProvider>
  );
}
