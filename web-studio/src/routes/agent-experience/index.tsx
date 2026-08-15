import * as React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import {
  BrainCircuitIcon,
  EyeIcon,
  LoaderCircleIcon,
  MessageSquareTextIcon,
  RefreshCwIcon,
  RouteIcon,
  SearchIcon,
  XIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Card } from '#/components/ui/card'
import { Input } from '#/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { useAppConnection } from '#/hooks/use-app-connection'
import { isOvClientError } from '#/lib/ov-client'

import { EvolutionSettingsPopover } from './-components/evolution-settings-popover'
import { ExperiencePreviewSheet } from './-components/experience-preview-sheet'
import { useExperienceTrajectoryTotal } from './-hooks/use-experience-trajectory-total'
import { fetchExperiences } from './-lib/api'
import {
  buildExperiencesUri,
  formatTimestamp,
  isExperienceUpdatedSinceLastSeen,
  markExperiencesSeen,
} from './-lib/experience'
import type { ExperienceFileItem } from './-lib/types'

export const Route = createFileRoute('/agent-experience/')({
  component: AgentExperienceRoute,
})

function getErrorMessage(error: unknown): string {
  if (isOvClientError(error) || error instanceof Error) {
    return error.message
  }
  return String(error)
}

function HighlightedText({ keyword, text }: { keyword: string; text: string }) {
  const normalizedKeyword = keyword.trim().toLocaleLowerCase()
  if (!normalizedKeyword) return <>{text}</>

  const normalizedText = text.toLocaleLowerCase()
  const fragments: React.ReactNode[] = []
  let cursor = 0
  let matchIndex = normalizedText.indexOf(normalizedKeyword)

  while (matchIndex !== -1) {
    if (matchIndex > cursor) {
      fragments.push(text.slice(cursor, matchIndex))
    }
    const matchEnd = matchIndex + normalizedKeyword.length
    fragments.push(
      <mark
        key={`${matchIndex}-${matchEnd}`}
        className="rounded-xs bg-primary/15 px-0.5 text-inherit"
      >
        {text.slice(matchIndex, matchEnd)}
      </mark>,
    )
    cursor = matchEnd
    matchIndex = normalizedText.indexOf(normalizedKeyword, cursor)
  }
  if (cursor < text.length) {
    fragments.push(text.slice(cursor))
  }
  return <>{fragments}</>
}

function EmptyHelpChecklist() {
  const { t } = useTranslation('agentExperiencePage')
  const reasons = [
    t('help.reasonConnected'),
    t('help.reasonSessions'),
    t('help.reasonCommit'),
  ]

  return (
    <div className="grid max-w-md gap-2 rounded-lg border border-dashed bg-muted/30 px-4 py-3 text-left">
      <p className="text-sm text-muted-foreground">{t('help.title')}</p>
      <ol className="grid gap-1.5 text-sm text-muted-foreground">
        {reasons.map((reason, index) => (
          <li className="flex items-center gap-2" key={reason}>
            <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-medium text-muted-foreground">
              {index + 1}
            </span>
            {reason}
          </li>
        ))}
      </ol>
    </div>
  )
}

function TrajectoryTotalCell({
  experienceUri,
  identityScopeKey,
}: {
  experienceUri: string
  identityScopeKey: string
}) {
  const { t } = useTranslation('agentExperiencePage')
  const { isPending, total } = useExperienceTrajectoryTotal(
    experienceUri,
    identityScopeKey,
  )

  if (isPending) {
    return (
      <span className="flex items-center gap-1 text-xs text-muted-foreground">
        <LoaderCircleIcon className="size-3 animate-spin" />
      </span>
    )
  }
  if (total === undefined) return <span className="text-xs">-</span>
  if (total === 0)
    return <span className="text-xs text-muted-foreground">0</span>

  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-primary">
      <RouteIcon className="size-3" />
      {t('detail.totalApplied', { count: total })}
    </span>
  )
}

function AgentExperienceRoute() {
  const { t, i18n } = useTranslation('agentExperiencePage')
  const { connection, identityScopeKey } = useAppConnection()
  const [keyword, setKeyword] = React.useState('')
  const [previewExperience, setPreviewExperience] =
    React.useState<ExperienceFileItem | null>(null)

  const experiencesUri = buildExperiencesUri(connection.userId)
  const experiencesQuery = useQuery({
    queryFn: ({ signal }) => fetchExperiences(experiencesUri, signal),
    queryKey: ['agent-experience-list', identityScopeKey, experiencesUri],
    staleTime: 30_000,
  })

  const experiences = experiencesQuery.data ?? []
  const normalizedKeyword = keyword.trim().toLocaleLowerCase()
  const visibleExperiences = React.useMemo(() => {
    if (!normalizedKeyword) return experiences
    return experiences.filter(
      (experience) =>
        experience.name.toLocaleLowerCase().includes(normalizedKeyword) ||
        experience.uri.toLocaleLowerCase().includes(normalizedKeyword),
    )
  }, [experiences, normalizedKeyword])

  // Snapshot "updated since last visit" badges when the list settles, then
  // mark the whole list as seen. Comparing against the pre-visit snapshot
  // (instead of live state) keeps badges visible for the current visit.
  const [updatedUris, setUpdatedUris] = React.useState<ReadonlySet<string>>(
    () => new Set(),
  )
  const markedRef = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (!experiencesQuery.isSuccess || experiences.length === 0) return
    const fingerprint = `${experiencesUri}:${experiences
      .map((item) => item.uri)
      .join('|')}`
    if (markedRef.current === fingerprint) return
    markedRef.current = fingerprint

    setUpdatedUris(
      new Set(
        experiences
          .filter((experience) =>
            isExperienceUpdatedSinceLastSeen(
              experience.uri,
              experience.modTime,
            ),
          )
          .map((experience) => experience.uri),
      ),
    )
    markExperiencesSeen(experiences)
  }, [experiences, experiencesQuery.isSuccess, experiencesUri])

  const connectionUnavailable =
    isOvClientError(experiencesQuery.error) &&
    experiencesQuery.error.code === 'NETWORK_ERROR'

  const handleOpenPreview = (experience: ExperienceFileItem) => {
    setPreviewExperience(experience)
  }

  return (
    <div className="flex w-full min-w-0 flex-col gap-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1.5">
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-semibold tracking-tight">
              {t('title')}
            </h1>
            {experiences.length > 0 ? (
              <Badge variant="outline" className="font-normal">
                {experiences.length}
              </Badge>
            ) : null}
          </div>
          <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
            {t('description')}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <EvolutionSettingsPopover />
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={experiencesQuery.isFetching}
            onClick={() => void experiencesQuery.refetch()}
          >
            <RefreshCwIcon
              className={
                experiencesQuery.isFetching ? 'animate-spin' : undefined
              }
            />
            {t('refresh')}
          </Button>
        </div>
      </header>

      {experiencesQuery.isLoading ? (
        <Card className="min-h-56 items-center justify-center">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderCircleIcon className="size-4 animate-spin" />
            {t('loading')}
          </div>
        </Card>
      ) : experiencesQuery.isError ? (
        <Card className="min-h-56 items-center justify-center px-6 text-center">
          <div className="grid gap-1">
            <p className="font-medium">{t('loadFailed')}</p>
            <p className="max-w-xl text-sm text-muted-foreground">
              {connectionUnavailable
                ? t('networkError')
                : getErrorMessage(experiencesQuery.error)}
            </p>
            {connectionUnavailable ? (
              <Button
                render={<Link to="/settings" />}
                nativeButton={false}
                variant="outline"
                size="sm"
                className="mx-auto mt-2"
              >
                {t('connectionSettings')}
              </Button>
            ) : (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="mx-auto mt-2"
                disabled={experiencesQuery.isFetching}
                onClick={() => void experiencesQuery.refetch()}
              >
                <RefreshCwIcon
                  className={
                    experiencesQuery.isFetching ? 'animate-spin' : undefined
                  }
                />
                {t('refresh')}
              </Button>
            )}
          </div>
        </Card>
      ) : experiences.length === 0 ? (
        <Card className="min-h-56 items-center justify-center px-6 text-center">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <BrainCircuitIcon className="size-5" />
          </div>
          <div className="grid max-w-md gap-1">
            <p className="font-medium">{t('empty')}</p>
            <p className="text-sm text-muted-foreground">
              {t('emptyDescription')}
            </p>
            <Button
              render={<Link to="/sessions" />}
              nativeButton={false}
              variant="outline"
              size="sm"
              className="mx-auto mt-3"
            >
              <MessageSquareTextIcon />
              {t('emptyAction')}
            </Button>
          </div>
          <div className="pt-4">
            <EmptyHelpChecklist />
          </div>
        </Card>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative w-full max-w-sm">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                aria-label={t('searchPlaceholder')}
                autoComplete="off"
                className="pl-8"
                name="agent-experience-search"
                placeholder={t('searchPlaceholder')}
                value={keyword}
                onChange={(event) => setKeyword(event.target.value)}
              />
              {keyword ? (
                <button
                  type="button"
                  aria-label={t('searchClear')}
                  className="absolute top-1/2 right-2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  onClick={() => setKeyword('')}
                >
                  <XIcon className="size-3.5" />
                </button>
              ) : null}
            </div>
            <Badge variant="outline" className="gap-1 font-normal">
              {t('directoryHint')}
            </Badge>
          </div>

          <Card size="sm" className="px-0">
            {visibleExperiences.length === 0 ? (
              <div className="grid min-h-40 place-items-center px-6 py-8 text-center">
                <div className="grid max-w-md gap-1">
                  <p className="font-medium">{t('searchNoResults')}</p>
                  <p className="text-sm text-muted-foreground">
                    {t('searchNoResultsDescription')}
                  </p>
                </div>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="pl-5">{t('columnFile')}</TableHead>
                    <TableHead className="w-36">{t('columnApplied')}</TableHead>
                    <TableHead className="w-44">{t('columnUpdated')}</TableHead>
                    <TableHead className="w-28 pr-5 text-right">
                      {t('columnActions')}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visibleExperiences.map((experience) => {
                    const updated = formatTimestamp(
                      experience.modTime,
                      i18n.language,
                    )
                    const isUpdated = updatedUris.has(experience.uri)

                    return (
                      <TableRow
                        key={experience.uri}
                        className="cursor-pointer"
                        onClick={() => handleOpenPreview(experience)}
                      >
                        <TableCell className="max-w-0 pl-5">
                          <div className="grid min-w-0 gap-0.5">
                            <div className="flex min-w-0 items-center gap-1.5">
                              <button
                                type="button"
                                className="min-w-0 truncate text-left font-medium underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                                onClick={(event) => {
                                  event.stopPropagation()
                                  handleOpenPreview(experience)
                                }}
                              >
                                <HighlightedText
                                  keyword={normalizedKeyword}
                                  text={experience.name}
                                />
                              </button>
                              {isUpdated ? (
                                <Badge
                                  variant="secondary"
                                  className="h-4 shrink-0 px-1 text-[10px]"
                                >
                                  {t('updatedBadge')}
                                </Badge>
                              ) : null}
                            </div>
                            <span className="truncate font-mono text-xs text-muted-foreground">
                              <HighlightedText
                                keyword={normalizedKeyword}
                                text={experience.uri}
                              />
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="w-36">
                          <TrajectoryTotalCell
                            experienceUri={experience.uri}
                            identityScopeKey={identityScopeKey}
                          />
                        </TableCell>
                        <TableCell className="w-44 text-sm text-muted-foreground">
                          {updated ? t('updated', { time: updated }) : '-'}
                        </TableCell>
                        <TableCell
                          className="w-28 pr-5 text-right"
                          onClick={(event) => event.stopPropagation()}
                        >
                          <Button
                            render={
                              <Link
                                params={{ experienceUri: experience.uri }}
                                to="/agent-experience/$experienceUri"
                              />
                            }
                            nativeButton={false}
                            size="xs"
                            variant="outline"
                            aria-label={t('openDetail', {
                              name: experience.name,
                            })}
                          >
                            <EyeIcon className="size-3.5" />
                            {t('viewAnalysis')}
                          </Button>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            )}
          </Card>
        </>
      )}

      <ExperiencePreviewSheet
        experience={previewExperience}
        language={i18n.language}
        onClose={() => setPreviewExperience(null)}
      />
    </div>
  )
}
