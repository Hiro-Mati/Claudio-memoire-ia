import * as React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import {
  BrainCircuitIcon,
  ChevronRightIcon,
  LoaderCircleIcon,
  MessageSquareTextIcon,
  RefreshCwIcon,
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

import { fetchExperiences } from './-lib/api'
import {
  buildExperiencesUri,
  formatTimestamp,
  getExperienceDisplayName,
} from './-lib/experience'

export const Route = createFileRoute('/agent-experience/')({
  component: AgentExperienceRoute,
})

function getErrorMessage(error: unknown): string {
  if (isOvClientError(error) || error instanceof Error) {
    return error.message
  }
  return String(error)
}

function AgentExperienceRoute() {
  const { t, i18n } = useTranslation('agentExperiencePage')
  const { connection, identityScopeKey } = useAppConnection()
  const [keyword, setKeyword] = React.useState('')

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

  const connectionUnavailable =
    isOvClientError(experiencesQuery.error) &&
    experiencesQuery.error.code === 'NETWORK_ERROR'

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
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={experiencesQuery.isFetching}
          onClick={() => void experiencesQuery.refetch()}
        >
          <RefreshCwIcon
            className={experiencesQuery.isFetching ? 'animate-spin' : undefined}
          />
          {t('refresh')}
        </Button>
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
                    <TableHead className="w-44">{t('columnUpdated')}</TableHead>
                    <TableHead className="w-28 text-right pr-5">
                      {t('columnActions')}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visibleExperiences.map((experience) => {
                    const displayName = getExperienceDisplayName(experience.uri)
                    const updated = formatTimestamp(
                      experience.modTime,
                      i18n.language,
                    )

                    return (
                      <TableRow key={experience.uri} className="cursor-pointer">
                        <TableCell className="max-w-0 pl-5">
                          <Link
                            className="grid min-w-0 gap-0.5 outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                            params={{ experienceUri: experience.uri }}
                            to="/agent-experience/$experienceUri"
                          >
                            <span className="truncate font-medium underline-offset-2 hover:underline">
                              {experience.name}
                            </span>
                            <span className="truncate font-mono text-xs text-muted-foreground">
                              {experience.uri}
                            </span>
                          </Link>
                        </TableCell>
                        <TableCell className="w-44 text-sm text-muted-foreground">
                          {updated
                            ? t('updated', { time: updated })
                            : displayName}
                        </TableCell>
                        <TableCell className="w-28 pr-5 text-right">
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
                          >
                            {t('viewAnalysis')}
                            <ChevronRightIcon className="size-3.5" />
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
    </div>
  )
}
