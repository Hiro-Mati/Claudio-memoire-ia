import { getOvResult, isOvClientError, ovClient } from '#/lib/ov-client'

import {
  DEFAULT_TRAJECTORY_PAGE_SIZE,
  normalizeExperienceFiles,
  normalizeOutcomeDistribution,
  normalizeSourceTrajectoryLinks,
  normalizeTrajectoryPage,
} from './experience'
import type { SourceTrajectoryLink } from './experience'
import type {
  AgentEvolutionStatus,
  ExperienceFileItem,
  OutcomeDistribution,
  TimeRange,
  TrajectoryPage,
} from './types'

/**
 * List experience memory files under `viking://user/<userId>/memories/experiences`.
 *
 * A 404 means the directory has not been created yet (no experience has ever
 * been extracted), which is surfaced as an empty list instead of an error.
 */
export async function fetchExperiences(
  experiencesUri: string,
  signal?: AbortSignal,
): Promise<ExperienceFileItem[]> {
  try {
    const result = await getOvResult<unknown>(
      ovClient.client.get({
        query: {
          node_limit: 1000,
          output: 'original',
          sort_by: 'mtime',
          sort_order: 'desc',
          uri: experiencesUri,
        },
        signal,
        url: '/api/v1/fs/ls',
      }),
    )
    return normalizeExperienceFiles(result)
  } catch (error) {
    if (isOvClientError(error) && error.statusCode === 404) {
      return []
    }
    throw error
  }
}

/** Read the markdown content of an experience or trajectory file. */
export async function fetchContent(
  uri: string,
  signal?: AbortSignal,
): Promise<string> {
  const result = await getOvResult<unknown>(
    ovClient.client.get({
      query: {
        limit: -1,
        offset: 0,
        uri,
      },
      signal,
      url: '/api/v1/content/read',
    }),
  )

  if (typeof result === 'string') return result
  if (result && typeof result === 'object') {
    const record = result as Record<string, unknown>
    if (typeof record.content === 'string') return record.content
  }
  return ''
}

/**
 * Query trajectories produced by commits that read the experience
 * (`GET /api/v1/agent-evolution/experiences/trajectories`).
 */
export async function fetchTrajectories(options: {
  experienceUri: string
  limit?: number
  offset?: number
  timeRange?: TimeRange
  signal?: AbortSignal
}): Promise<TrajectoryPage> {
  const {
    experienceUri,
    limit = DEFAULT_TRAJECTORY_PAGE_SIZE,
    offset = 0,
    timeRange,
    signal,
  } = options

  const result = await getOvResult<unknown>(
    ovClient.client.get({
      query: {
        experience_uri: experienceUri,
        limit,
        offset,
        start_date: timeRange?.startDate,
        end_date: timeRange?.endDate,
      },
      signal,
      url: '/api/v1/agent-evolution/experiences/trajectories',
    }),
  )
  return (
    normalizeTrajectoryPage(result, experienceUri) ?? {
      experienceUri,
      items: [],
      total: 0,
      limit,
      offset,
      hasMore: false,
    }
  )
}

/**
 * Query the outcome distribution of trajectories that applied the experience
 * (`GET /api/v1/agent-evolution/experiences/outcomes`).
 */
export async function fetchOutcomeDistribution(options: {
  experienceUri: string
  timeRange?: TimeRange
  signal?: AbortSignal
}): Promise<OutcomeDistribution> {
  const { experienceUri, timeRange, signal } = options

  const result = await getOvResult<unknown>(
    ovClient.client.get({
      query: {
        experience_uri: experienceUri,
        start_date: timeRange?.startDate,
        end_date: timeRange?.endDate,
      },
      signal,
      url: '/api/v1/agent-evolution/experiences/outcomes',
    }),
  )
  return normalizeOutcomeDistribution(result, experienceUri)
}

/**
 * Read the Experience memory attributes and return the trajectories from
 * which it was derived.
 */
export async function fetchSourceTrajectories(
  uri: string,
  signal?: AbortSignal,
): Promise<SourceTrajectoryLink[]> {
  const result = await getOvResult<unknown>(
    ovClient.client.get({
      query: { uri },
      signal,
      url: '/api/v1/fs/attrs',
    }),
  )
  return normalizeSourceTrajectoryLinks(result)
}

/**
 * Read the deployment/account Agent Evolution switch
 * (`GET /api/v1/admin/agent-evolution`). Requires an admin or root API key.
 */
export async function fetchAgentEvolutionStatus(
  signal?: AbortSignal,
): Promise<AgentEvolutionStatus> {
  const result = await getOvResult<unknown>(
    ovClient.client.get({
      signal,
      url: '/api/v1/admin/agent-evolution',
    }),
  )
  const record =
    result && typeof result === 'object'
      ? (result as Record<string, unknown>)
      : {}
  return {
    enabled: record.enabled === true,
    accountId:
      typeof record.account_id === 'string' ? record.account_id : undefined,
  }
}

/**
 * Toggle the Agent Evolution switch
 * (`PUT /api/v1/admin/agent-evolution`). Requires an admin or root API key.
 */
export async function setAgentEvolutionEnabled(
  enabled: boolean,
): Promise<AgentEvolutionStatus> {
  const result = await getOvResult<unknown>(
    ovClient.client.put({
      body: { enabled },
      url: '/api/v1/admin/agent-evolution',
    }),
  )
  const record =
    result && typeof result === 'object'
      ? (result as Record<string, unknown>)
      : {}
  return {
    enabled: record.enabled === true,
    accountId:
      typeof record.account_id === 'string' ? record.account_id : undefined,
  }
}
