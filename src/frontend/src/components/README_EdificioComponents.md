# Componentes de Edifício

Este documento descreve como usar os componentes reutilizáveis para formulários de edifícios.

## Componentes Disponíveis

### 1. EdificioForm
Formulário completo para cadastro/edição de edifícios com fachadas.

### 2. EdificioModal
Modal que encapsula o EdificioForm para uso em popups.

## Como Usar

### Uso Básico - Formulário Inline

```jsx
import EdificioForm from '../components/EdificioForm';

const MinhaPage = () => {
  const [edificio, setEdificio] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (formData) => {
    setLoading(true);
    try {
      // Sua lógica de API aqui
      const response = await fetch('/api/edificios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      // Limpar formulário após sucesso
      setEdificio(null);
    } catch (err) {
      setError('Erro ao salvar');
    } finally {
      setLoading(false);
    }
  };

  return (
    <EdificioForm
      initialData={edificio}
      onSubmit={handleSubmit}
      onCancel={() => setEdificio(null)}
      loading={loading}
      error={error}
      isEditing={!!edificio}
    />
  );
};
```

### Uso Avançado - Modal

```jsx
import EdificioModal from '../components/EdificioModal';

const MinhaPage = () => {
  const [showModal, setShowModal] = useState(false);
  const [edificioEditando, setEdificioEditando] = useState(null);

  const handleModalSubmit = async (formData) => {
    // Sua lógica aqui
    console.log('Dados:', formData);
    setShowModal(false);
  };

  return (
    <>
      <button onClick={() => setShowModal(true)}>
        Novo Edifício
      </button>

      <EdificioModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        initialData={edificioEditando}
        onSubmit={handleModalSubmit}
        title="Edifício"
      />
    </>
  );
};
```

## Props do EdificioForm

| Prop | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `initialData` | Object | null | Dados iniciais para edição |
| `onSubmit` | Function | - | Callback ao submeter (recebe formData) |
| `onCancel` | Function | - | Callback ao cancelar edição |
| `loading` | Boolean | false | Estado de carregamento |
| `error` | String | '' | Mensagem de erro |
| `isEditing` | Boolean | false | Se está em modo edição |
| `className` | String | '' | Classes CSS adicionais |

## Props do EdificioModal

| Prop | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `isOpen` | Boolean | - | Controla visibilidade do modal |
| `onClose` | Function | - | Callback para fechar modal |
| `title` | String | 'Edifício' | Título do modal |
| ... | ... | ... | + todas as props do EdificioForm |

## Estrutura dos Dados

### Input (initialData)
```javascript
{
  id: 1, // opcional, usado apenas para edição
  nome: 'Edifício Principal',
  localizacao: 'Centro da cidade',
  tipo: 'Comercial',
  pavimentos: 10,
  fachadas: [
    { area: 500, descricao: 'Fachada Norte' },
    { area: 450, descricao: 'Fachada Sul' }
  ]
}
```

### Output (onSubmit formData)
```javascript
{
  nome: 'Edifício Principal',
  localizacao: 'Centro da cidade',
  tipo: 'Comercial',
  pavimentos: 10, // Number
  fachadas: [
    { area: 500, descricao: 'Fachada Norte' },
    { area: 450, descricao: 'Fachada Sul' }
  ]
}
```

## Validações

O formulário inclui validações básicas:
- Todos os campos principais são obrigatórios
- Pavimentos deve ser um número >= 1
- Área das fachadas deve ser >= 0
- Não é possível adicionar fachada sem área e descrição

## Customização

### Estilos
Os componentes usam Tailwind CSS. Você pode:
- Passar `className` para sobrescrever estilos
- Modificar as classes diretamente no componente
- Usar CSS modules se preferir

### Campos Adicionais
Para adicionar novos campos:
1. Modifique o estado inicial em `EdificioForm`
2. Adicione o input no JSX
3. Atualize a validação se necessário

## Exemplo Completo

Veja `ExemploUsoEdificioForm.jsx` para exemplos práticos de uso.
