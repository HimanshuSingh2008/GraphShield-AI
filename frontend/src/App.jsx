import { useState } from 'react'
import './App.css'

const initialForm = {
  amount: '',
  step: '',
  oldbalanceOrg: '',
  newbalanceOrig: '',
  oldbalanceDest: '',
  newbalanceDest: '',
}

const fields = [
  ['amount', 'Transaction amount'],
  ['step', 'Step'],
  ['oldbalanceOrg', 'Sender balance before'],
  ['newbalanceOrig', 'Sender balance after'],
  ['oldbalanceDest', 'Receiver balance before'],
  ['newbalanceDest', 'Receiver balance after'],
]

export default function App() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function handleChange(event) {
    setForm({ ...form, [event.target.name]: event.target.value })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setResult(null)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(form).map(([key, value]) => [key, Number(value)]),
          ),
        ),
      })

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`)
      }

      setResult(await response.json())
    } catch {
      setError('Could not reach the API. Check that FastAPI is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <header className="app-header">
        <p className="eyebrow">GRAPHSHIELD AI · FRAUD DETECTION</p>
        <h1>Transaction Risk Check</h1>
        <p>Enter transaction details to estimate fraud risk.</p>
      </header>

      <form className="transaction-form" onSubmit={handleSubmit}>
        <div className="fields">
          {fields.map(([name, label]) => (
            <label key={name}>
              {label}
              <input
                name={name}
                type="number"
                min="0"
                step="any"
                value={form[name]}
                onChange={handleChange}
                required
              />
            </label>
          ))}
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Checking...' : 'Check transaction'}
        </button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <section className="result" aria-live="polite">
          <h2>{result.label}</h2>
          <p>
            Estimated fraud probability:{' '}
            <strong>{(result.fraud_probability * 100).toFixed(2)}%</strong>
          </p>
          <p>Decision threshold: {(result.threshold * 100).toFixed(0)}%</p>
        </section>
      )}
    </main>
  )
}
